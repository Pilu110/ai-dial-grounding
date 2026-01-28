import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient
from pydantic import RootModel

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)

# TODO: remove unused imports
# import asyncio
# from typing import Any, List, Dict
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
# from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# from pydantic import SecretStr, BaseModel, Field
# from task._constants import DIAL_URL, API_KEY
# from task.user_client import UserClient

SYSTEM_PROMPT = """You are a Hobbies Searching Wizard. Your job is to extract hobbies from user profiles and group user IDs by hobby.

## Instructions:
- You will receive a list of user profiles (user_id and about_me only) and a user request.
- Extract all hobbies mentioned in the user profiles that are relevant to the user request.
- For each hobby, list the user IDs of users who have that hobby.
- Output a JSON object in the following format:
  {
    "hobby1": [user_id1, user_id2, ...],
    "hobby2": [user_id3, ...]
  }
- Only use the provided context. Do not invent user data.
- Use the provided JSON schema for your response.
"""

USER_PROMPT = """## USER REQUEST:
{query}

## USER PROFILES:
{context}
"""

# --- Pydantic Models for Output Validation ---

class HobbyUsers(RootModel[dict[str, list[str]]]):
    """A mapping from hobby name to a list of user IDs."""
    pass

# --- Main Wizard Class ---

def format_user_minimal(user: dict[str, Any]) -> str:
    return f"user_id: {user['id']}, about_me: {user['about_me']}"

class HobbiesWizard:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.embeddings = embeddings
        self.llm_client = llm_client
        self.vectorstore = None
        self.user_client = UserClient()

    async def __aenter__(self):
        print("🔎 Initializing vector store with current users...")
        users = self.user_client.get_all_users()
        documents = [
            Document(page_content=user['about_me'], metadata={'user_id': user['id']})
            for user in users
        ]
        self.vectorstore = await self._create_chroma_vectorstore(documents)
        print("✅ Vector store ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_chroma_vectorstore(self, documents: list[Document], batch_size: int = 100):
        chroma = Chroma(embedding_function=self.embeddings)
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            await chroma.aadd_documents(batch)
        return chroma

    async def update_vectorstore(self):
        users = self.user_client.get_all_users()
        user_ids = set(user['id'] for user in users)
        vector_ids = set(self.vectorstore.get()['ids'])
        new_users = [user for user in users if user['id'] not in vector_ids]
        deleted_ids = list(vector_ids - user_ids)
        if new_users:
            docs = [Document(page_content=user['about_me'], metadata={'user_id': user['id']}) for user in new_users]
            await self.vectorstore.aadd_documents(docs)
        if deleted_ids:
            await self.vectorstore.delete(ids=deleted_ids)

    async def retrieve_relevant_users(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        await self.update_vectorstore()
        results = self.vectorstore.similarity_search(query, k=k)
        return [ {"user_id": doc.metadata['user_id'], "about_me": doc.page_content} for doc in results ]

    def augment_prompt(self, query: str, user_context: list[dict[str, Any]]) -> str:
        context_str = "\n".join([f"user_id: {u['user_id']}, about_me: {u['about_me']}" for u in user_context])
        return USER_PROMPT.format(query=query, context=context_str)

    def extract_hobbies(self, augmented_prompt: str) -> HobbyUsers:
        # 1. Create PydanticOutputParser for HobbyUsers
        parser = PydanticOutputParser(pydantic_object=HobbyUsers)

        # 2. Create messages array with system and user prompt
        messages = [
            SystemMessagePromptTemplate.from_template(template=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]

        # 3. Generate prompt with format instructions
        prompt = ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())

        # 4. Invoke LLM and parse output
        hobby_users: HobbyUsers = (prompt | self.llm_client | parser).invoke({})
        return hobby_users

    async def output_grounding(self, hobby_to_user_ids: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
        result = {}
        for hobby, user_ids in hobby_to_user_ids.items():
            users_info = []
            for uid in user_ids:
                user = self.user_client.get_user_by_id(uid)
                if user:
                    users_info.append(user)
            result[hobby] = users_info
        return result

# --- Main Loop ---

async def main():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        model='text-embedding-3-small-1',
        dimensions=384
    )
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        model="gpt-4o"
    )
    try:
        async with HobbiesWizard(embeddings, llm_client) as wizard:
            print("Sample queries:")
            print(" - I need people who love to go to mountains")
            print(" - Find users interested in painting and music")
            while True:
                user_query = input("> ").strip()
                if user_query.lower() in ['quit', 'exit']:
                    break
                user_context = await wizard.retrieve_relevant_users(user_query, k=20)
                augmented_prompt = wizard.augment_prompt(user_query, user_context)
                hobby_users = wizard.extract_hobbies(augmented_prompt)
                # hobby_users is a HobbyUsers object, so use .__root__ to get the dict
                result = await wizard.output_grounding(hobby_users.__root__)
                print(result)
    except Exception as ex:
        print("Failed to initialize Hobbies Searching Wizard.", ex)

asyncio.run(main())
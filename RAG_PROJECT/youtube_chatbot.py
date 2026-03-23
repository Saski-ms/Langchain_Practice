import logging
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv

# 🔹 Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

try:
    # 1️⃣ Get Transcript
    video_id = "gZkwoMyp3jc"
    logger.info(f"Fetching transcript for video: {video_id}")
    
    transcript = YouTubeTranscriptApi().fetch(video_id)
    logger.info(f"Transcript fetched successfully. Segments: {len(transcript)}")

    text = " ".join([t.text for t in transcript])
    logger.info(f"Transcript length (characters): {len(text)}")

    # 2️⃣ Split Text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    logger.info(f"Text split into {len(chunks)} chunks")

    # 3️⃣ Embeddings + Vector Store
    logger.info("Creating embeddings...")
    embeddings = OpenAIEmbeddings()

    logger.info("Storing embeddings in FAISS vector DB...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    logger.info("Vector store created successfully")

    # 4️⃣ Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info("Retriever initialized")

    # 5️⃣ Query
    question = "What is AI Agent"
    logger.info(f"User question: {question}")

    # 6️⃣ Retrieve relevant chunks
    docs = retriever.invoke(question)
    logger.info(f"Retrieved {len(docs)} relevant chunks")

    context_text = "\n".join([doc.page_content for doc in docs])
    logger.debug(f"Context preview: {context_text[:500]}")

    # 7️⃣ Prompt
    prompttemp = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    final_prompt = prompttemp.format(
        context=context_text,
        question=question
    )
    print(prompttemp)
    logger.info("Prompt created successfully")

    # 8️⃣ LLM
    logger.info("Sending request to LLM...")
    llm = ChatOpenAI(model="gpt-4o-mini")

    answer = llm.invoke(final_prompt)
    logger.info("Received response from LLM")

    print("\n✅ Answer:\n")
    print(answer.content)

except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
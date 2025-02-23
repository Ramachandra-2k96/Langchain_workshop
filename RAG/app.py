import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.llms import Ollama
import os
import shutil
import tempfile
from typing import List, Optional
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def load_documents(self, directory_path: Path) -> List[str]:
        """Load and process documents from the given directory."""
        try:
            # Load text documents
            text_loader = DirectoryLoader(
                str(directory_path),
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            
            # Load PDF documents
            pdf_loader = DirectoryLoader(
                str(directory_path),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            
            documents = []
            try:
                documents.extend(text_loader.load())
            except Exception as e:
                logger.error(f"Error loading text documents: {e}")
            
            try:
                documents.extend(pdf_loader.load())
            except Exception as e:
                logger.error(f"Error loading PDF documents: {e}")
            
            if not documents:
                raise ValueError("No documents were successfully loaded")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Concatenate chunks into a single context string
            context = "\n".join(chunk.page_content for chunk in chunks)
            return context
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            raise

class RAGApplication:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.initialize_llm()
        self.setup_prompt_template()
    
    def initialize_llm(self):
        """Initialize the language model with error handling."""
        try:
            self.llm = Ollama(model="llama3.2")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            st.error("Failed to initialize the language model. Please check if Ollama is running.")
            raise
    
    def setup_prompt_template(self):
        """Set up the RAG prompt template."""
        self.template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        
        If the answer cannot be found in the context, please respond with "I cannot find the answer in the provided documents."
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
    
    def save_uploaded_files(self, uploaded_files) -> bool:
        """Save uploaded files to temporary directory."""
        try:
            for file in uploaded_files:
                file_path = self.temp_dir / file.name
                file_path.write_bytes(file.getvalue())
            return True
        except Exception as e:
            logger.error(f"Error saving uploaded files: {e}")
            st.error("Failed to process uploaded files. Please try again.")
            return False
    
    def create_rag_chain(self, context: str):
        """Create the RAG chain with the given context."""
        try:
            return (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        except Exception as e:
            logger.error(f"Error creating RAG chain: {e}")
            st.error("Failed to create the processing chain. Please try again.")
            raise
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")

def main():
    try:
        # Initialize Streamlit app
        st.set_page_config(
            page_title="RAG Application with Llama 2",
            page_icon="📚",
            layout="wide"
        )
        
        st.title("RAG Application with Llama 2")
        
        # Initialize application
        app = RAGApplication()
        
        # File upload section
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload text or PDF files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        # Process uploaded files
        if uploaded_files:
            with st.spinner("Processing documents..."):
                if app.save_uploaded_files(uploaded_files):
                    try:
                        context = app.doc_processor.load_documents(app.temp_dir)
                        rag_chain = app.create_rag_chain(context)
                        
                        # Chat interface
                        st.subheader("Chat with your documents")
                        
                        # Initialize chat history in session state
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        
                        # Display chat history
                        for message in st.session_state.messages:
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])
                        
                        # Chat input
                        if prompt := st.chat_input("Ask a question about your documents"):
                            # Add user message to chat history
                            st.session_state.messages.append({
                                "role": "user",
                                "content": prompt,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            # Display user message
                            with st.chat_message("user"):
                                st.markdown(prompt)
                            
                            # Get response from chain
                            with st.spinner("Thinking..."):
                                try:
                                    response = rag_chain.invoke(prompt)
                                    
                                    # Add assistant message to chat history
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": response,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    
                                    # Display assistant message
                                    with st.chat_message("assistant"):
                                        st.markdown(response)
                                
                                except Exception as e:
                                    logger.error(f"Error generating response: {e}")
                                    st.error("Failed to generate response. Please try again.")
                    
                    except Exception as e:
                        logger.error(f"Error in document processing: {e}")
                        st.error("Failed to process documents. Please check your files and try again.")
        
        else:
            st.info("Please upload some documents to begin.")
        
        # Add a footer with system status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### System Status")
        st.sidebar.info("✅ System is running normally")
        
    except Exception as e:
        logger.error(f"Critical application error: {e}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")
    
    finally:
        # Cleanup temporary files
        if 'app' in locals():
            app.cleanup()

if __name__ == "__main__":
    main()
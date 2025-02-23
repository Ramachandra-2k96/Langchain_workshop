# Streamlit-based RAG (Retrieval-Augmented Generation) application using **Llama 3.2, LangChain, and document processing**:

---

## **Overview**
This script is a **Streamlit web application** that allows users to **upload documents (TXT/PDF), process them into smaller chunks**, and **ask questions** based on their content using a **Retrieval-Augmented Generation (RAG) model** powered by **Ollama (Llama 2)**.

---

## **1. Dependencies and Libraries**
The script relies on the following key Python libraries:

### **Core Libraries**
- `streamlit` → Creates the web interface.
- `os`, `shutil`, `tempfile`, `pathlib.Path` → Handles file operations and temporary storage.
- `logging` → Provides logging for debugging and error tracking.
- `datetime` → Used to timestamp messages in the chat history.

### **LangChain Components**
- `langchain_core.prompts.ChatPromptTemplate` → Defines a structured prompt for the LLM.
- `langchain_core.runnables.RunnablePassthrough` → Passes the input directly into the processing chain.
- `langchain_core.output_parsers.StrOutputParser` → Extracts plain text responses from the model.
- `langchain.text_splitter.RecursiveCharacterTextSplitter` → Splits long documents into smaller chunks for better processing.
- `langchain.document_loaders` (`DirectoryLoader`, `TextLoader`, `PyPDFLoader`) → Loads and processes different document formats.
- `langchain.llms.Ollama` → Connects to the Llama 2 model via **Ollama** for language processing.

---

## **2. Logging Configuration**
At the start of the script:
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```
- Sets up logging with **INFO level**.
- Logs messages in the format: `[Timestamp] - [Log Level] - [Message]`.

---

## **3. Document Processing (`DocumentProcessor` Class)**
This class handles **loading and processing** uploaded documents.

### **a. Initialization**
```python
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
```
- Uses **RecursiveCharacterTextSplitter** to **break documents into smaller chunks**.
- Each chunk is **1000 characters long**, with **200-character overlap** to preserve context.

---

### **b. Load Documents**
```python
def load_documents(self, directory_path: Path) -> List[str]:
```
- **Loads text (`.txt`) and PDF (`.pdf`) files** from the given directory.

#### **Text File Loading**
```python
text_loader = DirectoryLoader(
    str(directory_path),
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True
)
```
- Uses `DirectoryLoader` with `TextLoader` to **load all `.txt` files**.

#### **PDF File Loading**
```python
pdf_loader = DirectoryLoader(
    str(directory_path),
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)
```
- Uses `PyPDFLoader` to **extract text from PDFs**.

#### **Error Handling**
```python
try:
    documents.extend(text_loader.load())
except Exception as e:
    logger.error(f"Error loading text documents: {e}")
```
- **Catches errors** if loading fails and logs them.

#### **Splitting Documents**
```python
chunks = self.text_splitter.split_documents(documents)
context = "\n".join(chunk.page_content for chunk in chunks)
return context
```
- **Splits documents** into smaller chunks.
- **Concatenates chunks** into a **single context string**.

---

## **4. Application Class (`RAGApplication`)**
This class manages the entire **RAG pipeline**.

### **a. Initialization**
```python
class RAGApplication:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.initialize_llm()
        self.setup_prompt_template()
```
- **Creates a temporary directory** for uploaded files.
- **Initializes the Llama model**.
- **Sets up a structured prompt**.

---

### **b. Initialize the Llama Model**
```python
def initialize_llm(self):
    try:
        self.llm = Ollama(model="llama3.2")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        st.error("Failed to initialize the language model. Please check if Ollama is running.")
        raise
```
- Connects to the **Ollama Llama 2 model**.
- Displays an **error message** if the model fails to load.

---

### **c. Setup the RAG Prompt**
```python
def setup_prompt_template(self):
    self.template = """Answer the question based only on the following context:
    {context}

    Question: {question}

    If the answer cannot be found in the context, please respond with "I cannot find the answer in the provided documents."
    """
    self.prompt = ChatPromptTemplate.from_template(self.template)
```
- Defines a **structured prompt** to keep responses **context-relevant**.

---

### **d. Save Uploaded Files**
```python
def save_uploaded_files(self, uploaded_files) -> bool:
    try:
        for file in uploaded_files:
            file_path = self.temp_dir / file.name
            file_path.write_bytes(file.getvalue())
        return True
    except Exception as e:
        logger.error(f"Error saving uploaded files: {e}")
        st.error("Failed to process uploaded files. Please try again.")
        return False
```
- **Saves uploaded files** to the **temporary directory**.
- Logs any **errors in file saving**.

---

### **e. Create RAG Chain**
```python
def create_rag_chain(self, context: str):
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
```
- Uses **LangChain's Runnable pipeline** to:
  1. **Inject the context** into the template.
  2. **Pass user questions** to the model.
  3. **Parse the output** into plain text.

---

### **f. Cleanup Function**
```python
def cleanup(self):
    try:
        shutil.rmtree(self.temp_dir)
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {e}")
```
- **Deletes temporary files** after processing.

---

## **5. Streamlit Frontend (`main()` Function)**
This function sets up the **UI and logic**.

### **a. Streamlit App Setup**
```python
st.set_page_config(
    page_title="RAG Application with Llama 2",
    page_icon="📚",
    layout="wide"
)
```
- Configures the Streamlit page.

```python
st.title("RAG Application with Llama 2")
```
- Displays the **app title**.

---

### **b. File Upload Section**
```python
uploaded_files = st.file_uploader(
    "Upload text or PDF files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    key="file_uploader"
)
```
- Allows users to **upload multiple files**.

---

### **c. Process Documents and Enable Chat**
If documents are uploaded:
1. **Process them** into a context string.
2. **Create the RAG chain**.
3. **Enable the chat interface**.

```python
if uploaded_files:
    with st.spinner("Processing documents..."):
        if app.save_uploaded_files(uploaded_files):
            context = app.doc_processor.load_documents(app.temp_dir)
            rag_chain = app.create_rag_chain(context)
```

---

### **d. Chat Interface**
- Stores **chat history** in `st.session_state.messages`.
- Accepts **user input** and **displays responses**.

```python
if prompt := st.chat_input("Ask a question about your documents"):
    response = rag_chain.invoke(prompt)
```

---

## **Final Thoughts**
🚀 This **Streamlit-based RAG application** allows users to:
- **Upload documents**.
- **Chat with an AI** trained on those documents.
- **Get accurate, context-aware responses**.
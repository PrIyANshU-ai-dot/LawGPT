<h1 align="center">LawGPT - RAG based Generative AI Attorney Chatbot</h1>
<h3 align="center">Know Your Rights! Better Citizen, Better Nation!</h3>

<p align="center">
<img src="https://github.com/harshitv804/LawGPT/assets/100853494/ecff5d3c-f105-4ba2-a93a-500282f0bf00" width="700"/>
</p>

## About The Project
LawGPT is a RAG-based generative AI attorney chatbot that is trained using Indian Penal Code data.  
This project has been upgraded to use **Ollama’s local Llama 3 model** with **FAISS** for vector search — making it 100% local, private, and API-free.  
Ask any questions to the attorney and it will give you the right justice as per the IPC.  
Are you a noob in knowing your rights? Then this is for you! ⚖️
<br>

<div align="center">
  <br>
  <video src="https://github.com/harshitv804/LawGPT/assets/100853494/b6711fd6-87df-4a37-ba24-317c50dc6f8f" width="400" />
  <br>
</div>

### Check out the live demo on Hugging Face (legacy version using Together AI)  
<a href="https://huggingface.co/spaces/harshitv804/LawGPT">
  <img src="https://static.vecteezy.com/system/resources/previews/009/384/880/non_2x/click-here-button-clipart-design-illustration-free-png.png" width="120" height="auto">
</a>

## Getting Started

#### 1. Clone the repository:
   - ```
     git clone https://github.com/PrIyANshU-ai-dot/LawGPT.git
     cd LawGPT
     ```

#### 2. Set up a virtual environment:
   - ```
     python3 -m venv .venv
     source .venv/bin/activate      # macOS/Linux
     # OR
     .venv\Scripts\activate         # Windows
     ```

#### 3. Install necessary packages:
   - ```
     pip install -r requirements.txt
     ```
   - Or manually install the essentials:
     ```
     pip install streamlit langchain langchain-ollama langchain-community faiss-cpu pypdf
     ```

#### 4. Install and run Ollama:
   - Download Ollama from: [https://ollama.ai/download](https://ollama.ai/download)
   - Start the Ollama service:
     ```
     ollama serve
     ```
   - Pull the Llama 3 model:
     ```
     ollama pull llama3
     ```
   - Verify that it’s installed:
     ```
     ollama list
     ```

#### 5. Run the `ingest.py` file:
   - ```
     python ingest.py
     ```
   - This will:
     - Load all IPC PDFs from the `data` folder  
     - Split the text into smaller chunks  
     - Generate embeddings using `OllamaEmbeddings(model="llama3")`  
     - Create a FAISS vector database named `ipc_vector_db`

#### 6. Run the chatbot:
   - ```
     streamlit run app.py
     ```
   - Once running, open your browser and go to [http://localhost:8501](http://localhost:8501)

---

## Contact
If you have any questions or feedback, please raise an [GitHub issue]

# RAG - Retrieval Augmented Generation
Tutorial: [Video]([https://www.linkedin.com/posts/shaik-sirajuddin-144484243_ai-machinelearning-rag-activity-7211332250167042048-3s3W?utm_source=share&utm_medium=member_desktop])
## Overview

RAG leverages Meta Llama 3 (8B parameters) on GPU and Hugging Face API models on CPU. It supports two primary functionalities:

1. **Chat with LLM**: Engage in conversations with the large language model using GPU.
2. **RAG Chat with PDFs**: Perform Retrieval-Augmented Generation with up to 4 PDF documents.

## Features

- **Chat with LLM**: Utilize Meta Llama 3 for chat interactions, supporting system and user messages.
- **RAG Chat with PDFs**: Interact with content from PDFs using various prompts:
  - **Detailed Prompt**
  - **Short Prompt**
  - **Summary Prompt**
  - **Explanation Prompt**
  - **Opinion Prompt**
  - **Instruction Prompt**

## Setup and Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/SirajuddinShaik/RAG.git
   cd RAG
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   chainlit app.py
   ```

## Usage

To run the application with GPU support, you can use the following Colab link:
[Run on Colab](https://colab.research.google.com/drive/1Xgxrw3msyJZrwqJWuEipINv4k7yN2sTE?usp=sharing)

### Example Commands

- Start a chat session:

  ```bash
  chainlit app.py
  ```

- Load and interact with PDFs:
  Ensure your PDFs are in the appropriate directory and use the UI to upload and query them.

## Project Structure

- **app.py**: Main application script.
- **requirements.txt**: Dependencies required for the project.
- **src/**: Source code directory.
- **config/**: Configuration files.
- **src/utils/prompts**: various prompts used to interact.
- **data_ingestion/**: Scripts and tools for data ingestion.
- **logs/**: Log files.
- **Dockerfile**: Docker setup for containerized deployment.

## Contributing

We welcome contributions! Please fork the repository, create a new branch, and submit a pull request.

## License

This project is licensed under the MIT License.

---

If you have any questions or need further assistance, please open an issue on the GitHub repository.

---

Happy Coding! 🚀

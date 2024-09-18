# End-to-End Medical Chatbot Using FAISS and HuggingFace

## Project Overview

This project implements an end-to-end medical chatbot that leverages the FAISS library for efficient similarity search and the HuggingFace library for advanced embeddings. The chatbot allows users to ask medical-related questions, and it retrieves relevant responses from a provided medical book in PDF format.

## Features

- Load and process a medical book in PDF format.
- Split text into chunks for efficient searching.
- Use a pre-trained HuggingFace embedding model to convert text into vector embeddings.
- Index the embeddings using FAISS for fast similarity search.
- Simulate a chat session where users can ask questions and receive relevant answers.

## Installation

To run this project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/nitheesh2509/MedBot-Advanced-Medical-Question-Answering-System-Using-FAISS-and-HuggingFace.git
    cd <repository-directory>
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    Create a `requirements.txt` file if not already present and add the following packages:

    ```plaintext
    faiss-cpu==1.7.3
    numpy==1.24.3
    langchain==0.0.167
    HuggingFace-Embeddings==0.0.1
    PyPDF2==1.26.0
    ```

    Install the dependencies using:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your PDF file:**
   
   Place your medical book PDF in the `data` directory. Ensure the file is named `Medical_book.pdf` or update the file path in the script accordingly.

2. **Run the chatbot:**

    Execute the main script to start the chatbot:

    ```bash
    python chatbot.py
    ```

    You will be prompted to ask a medical question. Type your query and press Enter to receive a response. Type `exit` to quit the chatbot.

## Example

Here’s how the chatbot interaction might look:

Ask a medical question (or type 'exit' to quit): What is an arteriovenous fistula? Here are the relevant responses:

An arteriovenous fistula is an abnormal channel or passage between an artery and a vein.
Ask a medical question (or type 'exit' to quit): exit Exiting the chatbot. Have a great day!


## Troubleshooting

- **File Not Found Error:** Ensure that the PDF file is placed in the `data` directory and the file name matches the script’s expected name.
- **Dependency Issues:** Make sure all dependencies are installed correctly. Check for version conflicts, especially with NumPy.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact [your-email@example.com](mailto:nitheeshkumar2509@gmail.com).


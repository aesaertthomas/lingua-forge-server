# LinguaForge

🚀 **A FastAPI-based translation tool powered by MarianMT models from Hugging Face.**  

This project is an educational endeavor to explore deploying open-source machine learning models (specifically MarianMT for translation) using FastAPI. It serves as a learning experience for working with Hugging Face models, FastAPI, and deploying machine learning systems.  

While the project is still in development, it currently supports translation using MarianMT models and can be extended with a frontend or CLI interface in the future.  

---

## Features  
- **Translation using MarianMT**: Leverage Hugging Face's MarianMT models for fast and accurate translations.  
- **FastAPI Backend**: A lightweight and efficient API for handling translation requests.  
- **Educational Focus**: Learn how to deploy open-source models, use FastAPI, and integrate with Hugging Face.  
- **Extensible**: Designed to be extended with a frontend or CLI interface (future work).  

---

## Installation  

### Prerequisites  
Before installing, ensure you have the following installed:  
- Python 3.8+  
- CUDA (if using GPU acceleration)  
- PyTorch  
- Hugging Face Transformers  

### Steps  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/aesaertthomas/lingua-forge.git
   cd lingua-forge
# Project Documentation

## Install Dependencies (not yet implemented)
```bash
pip install -r requirements.txt
```
(Optional) Set up CUDA for GPU acceleration.

## Start the FastAPI Server
```bash
uvicorn main:app --reload
```

## Roadmap
Here’s what’s planned for the future of this project:

- Build a clean and user-friendly CLI interface.
- Add support for more languages using MarianMT models.
- Improve documentation and add usage examples.
- Explore frontend integration (if time permits 😅).

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

1. **Fork the repository**: Create your own copy of the project by forking it.
2. **Create a branch**: Make a new branch for your feature or bugfix.
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. **Make your changes**: Write your code, add documentation, or fix bugs.
4. **Test your changes**: Ensure everything works as expected.
5. **Submit a pull request**: Open a PR with a detailed description of your changes.

For major changes, please open an issue first to discuss what you’d like to add or change.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
This project wouldn’t be possible without the following resources and tools:

- Hugging Face for the MarianMT models and Transformers library.
- FastAPI for the awesome backend framework.
- The open-source community for inspiration and support.

## Contact
If you have any questions, feedback, or just want to chat about the project, feel free to reach out:

- **GitHub Issues**: Open an issue in this repository.
- **Email**: thomas.aesaert@gmail.com

## FAQs

**Q: Can I use this for production?**  
A: This project is primarily for educational purposes and may not be production-ready. However, feel free to adapt it for your needs!

**Q: How do I add support for more languages?**  
A: MarianMT supports many languages out of the box. Check the Hugging Face Model Hub for available models and update the code accordingly.

**Q: How can I contribute?**  
A: Check out the Contributing section for detailed steps on how to contribute to this project.

**Q: What if I encounter issues during installation?**  
A: Open an issue in the repository with details about the problem, and we’ll try to help you out!

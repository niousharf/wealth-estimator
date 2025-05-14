# Selfie Net Worth Estimator

This FastAPI-based application estimates an individual's net worth based on an image of their face. It leverages a pretrained ResNet-18 model to extract facial features (embeddings) and compares them to a set of mock wealthy profiles to find the most similar matches. 

This README includes the following:
- [Directory Structure](#directory-structure)
- [Architecture](#architecture)
- [Assumptions](#assumptions)
- [How to Run the Application Locally](#how-to-run-the-application-locally)
- [Dockerization](#dockerization)
- [Docker Hub Access](#docker-hub-access)
- [Testing the App in Docker](#testing-the-app-in-docker)
- [Public URI Testing](#public-uri-testing)


## Directory Structure
```bash
├── app
│   ├── __init__.py
│   ├── embeddings.py         # Embedding extraction logic
│   ├── main.py               # FastAPI app
│   ├── models.py             # Pretrained models and associated functions
│   ├── similarity.py         # Similarity calculation
│   └── templates             # HTML templates for rendering responses
├── Dockerfile                # Dockerfile for containerizing the app
├── mock_profiles
│   ├── wealthy_embeddings.npy   # Pre-generated embeddings for wealthy profiles
│   └── wealthy_metadata.json    # Metadata for matching profiles
├── requirements.txt          # Python dependencies
├── save_mock.py              # Script to generate mock embeddings and metadata
├── tests
│   ├── test_embeddings.py    # Tests for embedding extraction
│   ├── test_main.py          # Tests for the FastAPI app
│   ├── test_similarity.py    # Tests for similarity calculations
├── README.md                 # Project documentation
```

## Architecture

The app is designed with the following components:

1. **FastAPI App**: The main API server that handles image uploads and returns predictions. It uses FastAPI for easy-to-develop, asynchronous request handling.
2. **Model**: A pretrained ResNet-18 model (with its classification head removed) to extract image embeddings.
3. **Embedding Extraction**: The app processes the uploaded image, extracts the embedding vector, and compares it with mock embeddings of wealthy profiles.
4. **Similarity Calculation**: The app computes cosine similarity between the extracted image embeddings and stored mock embeddings to find the most similar profiles.
5. **Dockerization**: The app is containerized using Docker to facilitate easy deployment and testing in various environments.

### Assumptions

- **Mock Profiles**: The mock profiles of wealthy individuals are pre-generated and stored locally for testing purposes. These profiles are used to simulate the comparison and provide net worth estimates.
- **Embeddings**: The embeddings used for matching are stored in `wealthy_embeddings.npy` and the corresponding metadata in `wealthy_metadata.json`.
- **Image Input**: The app expects image input in a supported format (e.g., PNG, JPEG).

## How to Run the Application Locally

To run the app locally on your machine, follow the steps below:

### Prerequisites

- Python 3.8 or later
- Docker (optional, for containerized deployment)

### Install Requirements

1. Clone the repository or download the source code.
2. Navigate to the project folder and install dependencies:

```bash
pip install -r requirements.txt
```

### Run the FastAPI App Locally
To run the app locally without Docker:

1. In the project directory, run the FastAPI app using `uvicorn`:
```bash
uvicorn app.main:app --reload
```
2. The app will be accessible at `http://localhost:8000/`. You can test it by uploading an image of a wealthy person’s face to get a net worth estimate.


## Dockerization

This project is Dockerized for easy deployment and testing.

### Dockerfile Overview
The `Dockerfile` is designed to:

1. Install system dependencies (e.g., Python, dependencies).
2. Copy the application files into the container.
3. Install Python dependencies from requirements.txt.
4. Run the FastAPI app using Uvicorn on port 8000.

### Build and Run the Docker Container
1. **Build the Docker Image:**
In the project directory, run the following command to build the Docker image:
```bash
docker build -t wealth-estimator:<version> .
```
2. **Run the Docker Container:**
After building the image, you can run the Docker container with:
```bash
docker run -d -p 8000:8000 wealth-estimator:<version>
```
This will start the application on port 8000 inside the container, mapped to port 8000 on your machine.

3. **Access the App:**
Open your browser and go to `http://localhost:8000` to test the app locally inside the Docker container.

## Docker Hub Access
To use the Docker image from Docker Hub (assuming it's been pushed):

1. **Pull the Docker Image:**
If the image is published on Docker Hub, you can pull it using:
```bash
docker pull <your_docker_name>/wealth-estimator:<version>
```
A publicly accessible Docker image is available for testing on Docker Hub at the following address:
```bash
niousharf/wealth-estimator:2
```

2. **Run the Image:**
Once you've pulled the image, you can run it as explained above.

## Public URI Testing
Once the Docker container is running and deployed, you can access the application through a public URI. If the container is deployed on a cloud platform or has been exposed via a public domain, you can test the app from the public URL: [wealth-estimator.com](https://wealth-estimator.onrender.com)

**Important Note:** Since the application is hosted on a free tier of [render.com](https://render.com), it may take a few seconds to load. The server is put to sleep during periods of inactivity and needs time to wake up when accessed again.






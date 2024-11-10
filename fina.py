import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image 
from dotenv import load_dotenv
import io
import base64

# Load environment variables from .env file
load_dotenv()


class ImageProcessingAgent:
    """
    A class used to process images and store their embeddings in a Pinecone index.
    Attributes
    ----------
    index_name : str
        The name of the Pinecone index.
    model : CLIPModel
        The pre-trained CLIP model used for generating image embeddings.
    processor : CLIPProcessor
        The processor used for preparing images for the CLIP model.
    pinecone : Pinecone
        The Pinecone client instance.
    index : Pinecone.Index
        The Pinecone index instance.
    Methods
    -------
    __init__():
        Initializes the ImageProcessingAgent with a Pinecone index and CLIP model.
    create_index():
        Creates a Pinecone index if it does not already exist.
    process_images(image_files):
        Processes a list of image files, generates embeddings, and stores them in the Pinecone index.
    """

    def __init__(self):
        self.index_name = "suspect-identification"
        self.create_index()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def create_index(self):
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        try:
            index_names = self.pinecone.list_indexes().names()
        except:
            index_names = []
        if self.index_name not in index_names:
            self.pinecone.create_index(
                name=self.index_name,
                dimension=512,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = self.pinecone.Index(self.index_name)

    def process_images(self, image_files):
        for image_file in image_files:
            try:
                # Read image data
                image_bytes = image_file.read()
                image = Image.open(io.BytesIO(image_bytes))

                # Generate embedding
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_embedding = self.model.get_image_features(**inputs).numpy()[0]
                    normalized_embedding = image_embedding / torch.norm(
                        torch.tensor(image_embedding)
                    )

                # Convert image to base64 for storage
                buffered = io.BytesIO()
                image.save(buffered, format=image.format or "JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Store in Pinecone with image data as metadata
                self.index.upsert(
                    [
                        (
                            image_file.name,
                            normalized_embedding.tolist(),
                            {"image_data": img_str},
                        )
                    ]
                )

                st.success(f"Successfully processed {image_file.name}")

                # Reset file pointer for potential reuse
                image_file.seek(0)

            except Exception as e:
                st.error(f"Error processing image {image_file.name}: {str(e)}")


class TextEmbeddingAgent:
    """
    A class used to generate text embeddings using the CLIP model.
    Methods
    -------
    __init__():
        Initializes the TextEmbeddingAgent with a pre-trained CLIP model and processor.
    generate_embedding(description: str) -> list:
        Generates a normalized text embedding for the given description.
        Parameters:
            description (str): The text description to generate an embedding for.
        Returns:
            list: A list representing the normalized text embedding, or None if an error occurs.
    """

    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def generate_embedding(self, description):
        try:
            inputs = self.processor(
                text=[description], return_tensors="pt", padding=True
            )
            with torch.no_grad():
                text_embedding = self.model.get_text_features(**inputs).numpy()[0]
                normalized_embedding = text_embedding / torch.norm(
                    torch.tensor(text_embedding)
                )
            return normalized_embedding.tolist()
        except Exception as e:
            st.error(f"Error generating text embedding: {str(e)}")
            return None


class RetrievalAgent:
    """
    A class used to represent a Retrieval Agent that queries an index for matches based on a given text embedding.
    Attributes
    ----------
    index : object
        The index object that supports querying with a vector.
    Methods
    -------
    retrieve_matches(text_embedding, top_k=3)
        Queries the index with the provided text embedding and returns the top_k matches.
    """

    """
        Constructs all the necessary attributes for the Retrieval Agent object.
        Parameters
        ----------
        index : object
            The index object that supports querying with a vector.
        """
    """
        Queries the index with the provided text embedding and returns the top_k matches.
        Parameters
        ----------
        text_embedding : list or array-like
            The embedding of the text to query the index with.
        top_k : int, optional
            The number of top matches to return (default is 3).
        Returns
        -------
        list
            A list of the top_k matches from the index. If an error occurs, an empty list is returned.
        """

    def __init__(self, index):
        self.index = index

    def retrieve_matches(self, text_embedding, top_k=3):
        try:
            results = self.index.query(
                vector=text_embedding, top_k=top_k, include_metadata=True
            )
            return results["matches"]
        except Exception as e:
            st.error(f"Error retrieving matches: {str(e)}")
            return []


def display_image_from_base64(base64_str, caption):
    try:
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption=caption)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")


def main():
    st.title("Trace Vision - Suspect Identification System")
    st.write("Upload images and enter descriptions to find matching suspects.")

    # Create a session state to store the image agent
    if "image_agent" not in st.session_state:
        st.session_state.image_agent = ImageProcessingAgent()

    # Image upload section
    st.subheader("Upload Images")
    image_files = st.file_uploader(
        "Upload suspect images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Select one or more images to upload",
    )

    if st.button("Process and Store Images"):
        if image_files:
            with st.spinner("Processing images..."):
                st.session_state.image_agent.process_images(image_files)
        else:
            st.warning("Please upload some images first.")

    # Search section
    st.subheader("Search by Description")
    description = st.text_area(
        "Enter witness description",
        help="Describe the suspect's appearance, clothing, or other distinctive features",
    )

    if st.button("Search for Matches"):
        if description:
            with st.spinner("Searching for matches..."):
                text_agent = TextEmbeddingAgent()
                text_embedding = text_agent.generate_embedding(description)

                if text_embedding is not None:
                    retrieval_agent = RetrievalAgent(st.session_state.image_agent.index)
                    matched_images = retrieval_agent.retrieve_matches(text_embedding)

                    if matched_images:
                        st.subheader("Matching Results")
                        for idx, match in enumerate(matched_images, 1):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if (
                                    "metadata" in match
                                    and "image_data" in match["metadata"]
                                ):
                                    display_image_from_base64(
                                        match["metadata"]["image_data"], f"Match #{idx}"
                                    )
                                else:
                                    st.warning(f"No image data found for Match #{idx}")
                            with col2:
                                st.write(f"Confidence Score: {match['score']:.2%}")
                    else:
                        st.warning("No matching images found.")
        else:
            st.warning("Please enter a description to search.")


if __name__ == "__main__":
    main()

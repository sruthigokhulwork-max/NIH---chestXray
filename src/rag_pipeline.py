# src/rag_pipeline.py
import os
import sys
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import KB_PATH, BASE_DIR


# STEP 1: CHUNK THE KNOWLEDGE BASE FILE

def load_and_chunk_kb(kb_path=KB_PATH):
    
    # Reads thoracic_kb.txt and splits it into individual chunks,
    # one per [SECTION_NAME] marker.
    # WHY this approach: We control the file format ourselves (we wrote it), so splitting on our own [LABEL] markers is 
    # simple and 100% reliable — no need for complex NLP-based paragraph detection.
    # it returns: chunks : list of dicts, each with {"label": ..., "text": ...}

    with open(kb_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on lines like "[ATELECTASIS]" while keeping the marker WHY regex: \[([A-Z_]+)\] matches our exact bracket format
    content = "\n" + content
    sections = re.split(r"\n\[([A-Z_]+)\]\n", content)

    # adding leading newline ensures the FIRST [LABEL] marker behaves identically to all subsequent ones during splitting, 
    # avoiding edge-case bugs when the file starts directly with a marker.

    # re.split with a capturing group returns: ['', 'LABEL1', 'text1', 'LABEL2', 'text2', ...]
    # here,we skip the first empty string and pair up (label, text)
    chunks = []
    for i in range(1, len(sections), 2):
        label = sections[i].strip()
        text = sections[i + 1].strip()
        chunks.append({"label": label, "text": text})

    print(f"  Loaded {len(chunks)} knowledge base chunks")
    for chunk in chunks:
        print(f"     - {chunk['label']} ({len(chunk['text'])} chars)")

    return chunks


# STEP 2 & 3: EMBED CHUNKS AND BUILD FAISS INDEX

class KnowledgeBaseRetriever:

    # Wraps the embedding model + FAISS index together, so we can easily query "which knowledge base entries are most relevant
    # to these predicted diseases?"

    def __init__(self, kb_path=KB_PATH):
        print("BUILDING RAG KNOWLEDGE BASE INDEX")
     

        # Load chunks from our text file
        self.chunks = load_and_chunk_kb(kb_path)

        # WHY this specific model (all-MiniLM-L6-v2):Small, fast and widely used, well-tested for semantic similarity tasks
        print("\n  Loading sentence-transformer model...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Embed every chunk's text into a vector
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.embed_model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        print(f"  Generated embeddings: shape {embeddings.shape}")
        # shape = [15, 384] → 15 chunks, 384 numbers each

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        print(f"   FAISS index built with {self.index.ntotal} vectors")

    def retrieve(self, query_diseases, top_k=3):
  
        # Given a list of predicted disease names, retrieves the most
        # relevant knowledge base chunks.

        # Args:
        #     query_diseases : list of disease names, e.g. ["Edema", "Consolidation"]
        #     top_k          : how many chunks to retrieve PER disease query

        # Returns:
        #     retrieved_chunks : list of dicts (label + text), deduplicated

        # Combine disease names into a single search query because we want chunks relevant to the OVERALL prediction
        # set, not just one disease in isolation
        query_text = ", ".join(query_diseases)
        query_embedding = self.embed_model.encode([query_text]).astype("float32")

        # Search FAISS for the most similar chunks
        # distances = how far each result is (lower = more similar)
        # indices   = which chunk numbers matched
        distances, indices = self.index.search(query_embedding, top_k)

        retrieved = []
        for idx in indices[0]:
            retrieved.append(self.chunks[idx])

        return retrieved

# QUICK TEST
def test_rag_pipeline():
    print("\n🧪 TESTING RAG PIPELINE")
    print("=" * 60)

    retriever = KnowledgeBaseRetriever()

    # Simulate a model's predictions (like we saw from Grad-CAM testing)
    test_predictions = ["Edema", "Consolidation", "Pneumothorax"]

    print(f" Query (predicted diseases): {test_predictions}")
    print(f" Retrieving relevant knowledge base entries...")

    results = retriever.retrieve(test_predictions, top_k=3)

    print(f" Retrieved {len(results)} chunks:")
    for i, chunk in enumerate(results, 1):
        print(f"\n  [{i}] {chunk['label']}")
        print(f"      {chunk['text'][:150]}...")

    print(" RAG PIPELINE TEST COMPLETE!")


if __name__ == "__main__":
    test_rag_pipeline()
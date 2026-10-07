import chromadb


def test_department_filter():
    client = chromadb.Client()

    collection = client.create_collection(
        name="permission_test"
    )

    collection.add(
        ids=["eng1", "fin1", "hr1"],
        documents=[
            "Deployment architecture",
            "Annual company budget",
            "Employee leave policy"
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0]
        ],
        metadatas=[
            {"department": "Engineering"},
            {"department": "Finance"},
            {"department": "HR"}
        ]
    )

    results = collection.query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=3,
        where={
            "department": {
                "$in": ["Engineering"]
            }
        }
    )

    assert results["ids"][0] == ["eng1"]

    client.delete_collection("permission_test")
# Run Chromadb as Docker container
- Create dir to persist chromadb data files. Use anything for the dir.
  ```
   mkdir -p <local_persist_dir>
  ```
- Run docker with folder created above to mount as volume for persisting data files. Replace the `<local_persist_dir>` with where persist dir is created.
  ```
   docker run --name chromadb-container --mount type=bind,source=<local_persist_dir>,target=/data -p 8000:8000 chromadb/chroma
  ```

# Important Notes:
When creating embeddings, the container automatically loads the V6 model using the default embedding function; you can supply an alternate embedding model if needed.
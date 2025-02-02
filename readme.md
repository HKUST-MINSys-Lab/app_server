# Flask Server

A Flask-based REST API server that provides GPT model interaction and file management capabilities.

## Setup
1. Install dependencies:
```sh
pip install -r requirements.txt
```

2. Configure environment variables:
- Create `.env` file with:
  ```
  OPENAI_API_URL=your_openai_api_url
  OPENAI_API_KEY=your_openai_api_key
  DEEPSEEK_API_URL=your_deepseek_api_url
  DEEPSEEK_API_KEY=your_deepseek_api_key
  ```

3. Run the server with additional arguments:
```sh
python src/app.py --debug --host 0.0.0.0 --port 5000
```

see more [details](https://flask.palletsprojects.com/en/stable/tutorial/factory/#run-the-application)

## Features

### 1. GPT Integration
- **Endpoint**: `/ask`
- **Method**: POST
- **Request Format**:
  ```json
  {
    "prompt": "Your question here"
  }
  ```
- **Response**:
  ```json
  {
    "response": "GPT response content"
  }
  ```

### 2. File Management

#### Upload File
- **Endpoint**: `/upload`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameters**:
  - `file`: File to upload
  - `path`: Target directory path
- **Response**:
  ```json
  {
    "message": "File uploaded successfully"
  }
  ```

#### List Directory
- **Endpoint**: `/list`
- **Method**: GET
- **Query Parameters**:
  - `path`: Directory path to list
- **Response**:
  ```json
  ["file1.txt", "file2.txt", "directory1"]
  ```

#### Create Directory
- **Endpoint**: `/mkdir`
- **Method**: POST
- **Request Format**:
  ```json
  {
    "path": "desired/directory/path"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Directory created successfully"
  }
  ```

## Error Codes
- 410: No prompt provided
- 411: Unknown model specified
- 420: No file part in request
- 421: No path specified
- 422: Invalid path
- 423: Path does not exist
- 424: No selected file
- 425: Path already exists

## Todo
1. Add Database

## References
- [Flask File Uploading](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/)
- [Flask Database](https://flask.palletsprojects.com/en/stable/tutorial/database/)
- [Run application](https://flask.palletsprojects.com/en/stable/tutorial/factory/#run-the-application)
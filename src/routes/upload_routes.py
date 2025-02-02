from flask import Blueprint, request, jsonify
import os
from src.utils.err import (
    BaseErr,
    NoFilePartErr, 
    NoPathSpecifiedErr,
    InvalidPathErr, 
    PathDoesNotExistErr,
    NoSelectedFileErr,
    PathAlreadyExistsErr
)

upload_bp = Blueprint('upload', __name__)

root = './uploads'

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload a file to the specified path on the server.
    Request Format:
        - Content-Type: multipart/form-data
        - form fields:
            - file: The file to upload (required)
            - path: Target directory path on server (required)
    Returns:
        JSON response with:
            - success: {'message': 'File uploaded successfully'}, 200
            - errors: seen in src/utils/err.py
    Example:
        curl -X POST -F "file=@example.txt" -F "path=/target/directory" http://server/upload
    """
    try:
        if 'file' not in request.files:
            raise NoFilePartErr()
        
        path = request.form.get('path')
        if not path:
            raise NoPathSpecifiedErr()
            
        if '..' in path or path.startswith('/'):
            raise InvalidPathErr()
            
        base_path = root
        full_path = os.path.join(base_path, path.lstrip('/'))
        
        if not os.path.exists(full_path):
            raise PathDoesNotExistErr()
        
        file = request.files['file']
        if file.filename == '':
            raise NoSelectedFileErr()
            
        file.save(os.path.join(full_path, file.filename))
        return jsonify({'message': 'File uploaded successfully'}), 200
        
    except BaseErr as e:
        return jsonify({'error': e.message}), e.code

@upload_bp.route('/list', methods=['GET'])
def list_dir():
    """
    List contents of specified directory.
    Query parameters:
        - path: Target directory path to list (required)
    Returns:
        - JSON array of directory contents, 200
        - Error messages with 400 status for invalid paths
    """
    try:
        path = request.args.get('path')
        if not path:
            raise NoPathSpecifiedErr()
            
        if '..' in path or path.startswith('/'):
            raise InvalidPathErr()
            
        full_path = os.path.join(root, path.lstrip('/'))
        if not os.path.exists(full_path):
            raise PathDoesNotExistErr()
            
        contents = os.listdir(full_path)
        return jsonify(contents), 200
        
    except BaseErr as e:
        return jsonify({'error': e.message}), e.code

@upload_bp.route('/mkdir', methods=['POST'])
def mkdir():
    """
    Create a new directory.
    Request Format:
        - JSON with path field
    Returns:
        - Success message 200
        - Error messages with 400 status for invalid paths
    """
    try:
        path = request.json.get('path')
        if not path:
            raise NoPathSpecifiedErr()
            
        if '..' in path or path.startswith('/'):
            raise InvalidPathErr()
            
        full_path = os.path.join(root, path.lstrip('/'))
        if os.path.exists(full_path):
            raise PathAlreadyExistsErr()
            
        os.makedirs(full_path)
        return jsonify({'message': 'Directory created successfully'}), 200
        
    except BaseErr as e:
        return jsonify({'error': e.message}), e.code
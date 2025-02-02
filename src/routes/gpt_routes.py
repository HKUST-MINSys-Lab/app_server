from flask import Blueprint, request, jsonify
from src.services.query_gpt import query
from src.utils.err import BaseErr, NoPromptErr
gpt_bp = Blueprint('gpt', __name__)

@gpt_bp.route('/ask', methods=['POST'])
def ask_gpt():
    data = request.get_json()
    if not data or 'prompt' not in data:
        e = NoPromptErr()
        return jsonify({'error': e.message}), e.code

    try:
        content = query(data['prompt'], 'gpt-4o')
        return jsonify({'response': content}), 200
    except BaseErr as e:
        return jsonify({'error': e.message}), e.code
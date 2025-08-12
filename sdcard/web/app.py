from flask import Flask, jsonify, request, render_template
from sdcard.utils.cards import get_available_cards, import_cards
import click

app = Flask(__name__)

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/cards', methods=['GET'])
def list_cards():
    """API endpoint to list available SD cards"""
    format_type = request.args.get('format', 'exfat')
    cards = get_available_cards(format_type)
    return jsonify({
        'cards': cards,
        'count': len(cards)
    })

@app.route('/api/import', methods=['POST'])
def web_import_cards():
    """API endpoint to start import process"""
    data = request.get_json()
    selected_cards = data.get('cards', [])
    clean = data.get('clean', False)
    return import_cards(selected_cards, clean)

def run_server():
    app.run(debug=True)

if __name__ == '__main__':
    run_server()
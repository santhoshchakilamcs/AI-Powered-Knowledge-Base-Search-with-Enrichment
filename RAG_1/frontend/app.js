// API Base URL
const API_BASE = '/api';

// Global state
let currentAnswer = null;
let currentQuery = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadDocuments();
    checkHealth();
});

// Check API health
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API health check failed:', error);
    }
}

// Upload document
async function uploadDocument() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.innerHTML = '<div class="loading">⏳ Uploading and processing...</div>';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE}/documents/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const data = await response.json();
        statusDiv.innerHTML = `
            <div style="color: #4caf50; padding: 10px; background: #e8f5e9; border-radius: 6px;">
                ✅ Successfully uploaded! Created ${data.chunks_created} chunks.
            </div>
        `;
        
        // Reload documents list
        loadDocuments();
        
        // Clear file input
        fileInput.value = '';
        
        // Clear status after 3 seconds
        setTimeout(() => {
            statusDiv.innerHTML = '';
        }, 3000);
        
    } catch (error) {
        statusDiv.innerHTML = `
            <div style="color: #f44336; padding: 10px; background: #ffebee; border-radius: 6px;">
                ❌ Upload failed: ${error.message}
            </div>
        `;
    }
}

// Load documents list
async function loadDocuments() {
    const listDiv = document.getElementById('documentsList');
    const countSpan = document.getElementById('docCount');
    
    try {
        const response = await fetch(`${API_BASE}/documents`);
        const data = await response.json();
        
        countSpan.textContent = data.total_count;
        
        if (data.documents.length === 0) {
            listDiv.innerHTML = '<p style="color: #666; text-align: center;">No documents yet</p>';
            return;
        }
        
        listDiv.innerHTML = data.documents.map(doc => `
            <div class="document-item">
                <div>
                    <strong>${doc.filename}</strong><br>
                    <small style="color: #666;">
                        ${formatFileSize(doc.file_size)} • Uploaded ${new Date(doc.upload_timestamp * 1000).toLocaleDateString()}
                    </small>
                </div>
                <button class="delete-btn" onclick="deleteDocument('${encodeURIComponent(doc.filename)}')">
                    Delete
                </button>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error loading documents:', error);
        listDiv.innerHTML = '<p style="color: #f44336;">Error loading documents</p>';
    }
}

// Delete document
async function deleteDocument(filename) {
    if (!confirm(`Are you sure you want to delete "${decodeURIComponent(filename)}"?`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/documents/${filename}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Delete failed with status ${response.status}`);
        }

        const result = await response.json();
        console.log('Delete successful:', result);

        // Refresh the document list
        loadDocuments();

        // Show success message
        alert('Document deleted successfully!');

    } catch (error) {
        console.error('Delete error:', error);
        alert('Failed to delete document: ' + error.message);
    }
}

// Search knowledge base
async function searchKnowledgeBase() {
    const query = document.getElementById('searchQuery').value.trim();
    const resultsDiv = document.getElementById('results');

    if (!query) {
        alert('Please enter a question');
        return;
    }

    currentQuery = query;
    resultsDiv.innerHTML = '<div class="loading">🤔 Thinking...</div>';

    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                enable_auto_enrichment: true  // Always enabled - auto-enriches only when needed
            })
        });
        
        if (!response.ok) {
            throw new Error('Search failed');
        }
        
        const data = await response.json();
        currentAnswer = data.answer;
        displayResults(data);
        
    } catch (error) {
        resultsDiv.innerHTML = `
            <div style="color: #f44336; padding: 15px;">
                ❌ Search failed: ${error.message}
            </div>
        `;
    }
}

// Display search results
function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    
    const confidencePercent = (data.confidence * 100).toFixed(0);
    const confidenceBadge = data.confidence >= 0.7 ? 'badge-success' : 
                           data.confidence >= 0.4 ? 'badge-warning' : 'badge-danger';
    
    const completeBadge = data.is_complete ? 
        '<span class="badge badge-success">Complete</span>' : 
        '<span class="badge badge-warning">Incomplete</span>';
    
    let html = `
        <div class="answer-section">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="color: #c7d2fe; font-size: 1.3em;">Answer</h3>
                <div>
                    ${completeBadge}
                    <span class="badge ${confidenceBadge}">${confidencePercent}% Confident</span>
                </div>
            </div>

            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
            </div>

            <p style="margin-top: 15px; line-height: 1.8; color: #e4e4e7; font-size: 1.05em;">${data.answer}</p>
    `;
    
    // Auto-enrichment info
    if (data.auto_enrichment_applied) {
        html += `
            <div style="margin-top: 15px; padding: 16px; background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.15)); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 10px; color: #a5b4fc; font-size: 0.95em;">
                🌐 <strong style="color: #c7d2fe;">Auto-enriched</strong> with information from: <span style="color: #e0e7ff;">${data.auto_enrichment_sources.join(', ')}</span>
            </div>
        `;
    }
    
    // Sources
    if (data.sources && data.sources.length > 0) {
        html += `
            <div class="sources">
                <h4 style="color: #c7d2fe; margin-bottom: 10px;">📚 Sources</h4>
                ${data.sources.map(source => {
                    // Safety check for source object
                    if (!source) return '';

                    const isExternal = source.metadata && source.metadata.enriched;
                    const sourceUrl = source.metadata && source.metadata.url;
                    const sourceType = source.metadata && source.metadata.source;

                    // Handle different source formats
                    const documentName = source.document_name || source.filename || 'Unknown Document';
                    const relevanceScore = source.relevance_score || 0.8; // Default score
                    const isUploaded = source.type === 'uploaded';

                    return `
                        <div class="source-item">
                            <strong style="color: #e4e4e7;">
                                ${isExternal ? '🌐 ' : (isUploaded ? '📄 ' : '📄 ')}${documentName}
                            </strong>
                            <span class="badge badge-success" style="margin-left: 10px;">
                                ${(relevanceScore * 100).toFixed(0)}% relevant
                            </span>
                            ${isExternal ? `
                                <span class="badge badge-warning" style="margin-left: 5px;">
                                    External Source
                                </span>
                            ` : isUploaded ? `
                                <span class="badge badge-info" style="margin-left: 5px;">
                                    Uploaded Document
                                </span>
                            ` : ''}
                            <p style="margin-top: 8px; color: #a1a1aa; font-size: 0.9em; line-height: 1.5;">
                                ${source && source.content ? source.content.substring(0, 200) + '...' :
                                  source && source.filename ? `Document: ${source.filename}` :
                                  'Source content not available'}
                            </p>
                            ${sourceUrl ? `
                                <a href="${sourceUrl}" target="_blank" rel="noopener noreferrer"
                                   style="color: #60a5fa; text-decoration: none; font-size: 0.85em; margin-top: 6px; display: inline-block;">
                                    🔗 View Full Source
                                </a>
                            ` : ''}
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }
    
    // Missing info and enrichment suggestions
    if (!data.is_complete && data.missing_info && data.missing_info.length > 0) {
        html += `
            <div class="enrichment-suggestions">
                <h4 style="color: #fbbf24; margin-bottom: 10px;">💡 Missing Information</h4>
                <ul style="margin-left: 20px; color: #e4e4e7;">
                    ${data.missing_info.map(info => `<li style="margin: 8px 0;">${info}</li>`).join('')}
                </ul>
        `;

        if (data.enrichment_suggestions && data.enrichment_suggestions.length > 0) {
            html += `
                <h4 style="color: #fbbf24; margin-top: 15px; margin-bottom: 10px;">
                    🎯 Enrichment Suggestions
                </h4>
                ${data.enrichment_suggestions.map(suggestion => `
                    <div class="suggestion-item">
                        <strong style="color: #c7d2fe;">${suggestion.type}</strong> <span style="color: #a1a1aa;">(${suggestion.priority} priority)</span><br>
                        <span style="color: #d4d4d8; margin-top: 4px; display: block;">${suggestion.suggestion}</span>
                        ${suggestion.external_source_url ? `
                            <br>
                            <a href="${suggestion.external_source_url}" target="_blank" rel="noopener noreferrer"
                               style="color: #60a5fa; text-decoration: none; font-size: 0.9em; margin-top: 6px; display: inline-block;">
                                🔗 View Source
                            </a>
                        ` : ''}
                    </div>
                `).join('')}
            `;
        }

        html += `</div>`;
    }
    
    // Rating section
    html += `
        <div class="rating-section">
            <h4 style="color: #c7d2fe; margin-bottom: 10px;">⭐ Rate this answer</h4>
            <div class="stars" id="ratingStars">
                ${[1, 2, 3, 4, 5].map(i => `
                    <span class="star" data-rating="${i}" onclick="rateAnswer(${i})">★</span>
                `).join('')}
            </div>
            <div id="ratingFeedback" style="margin-top: 10px;"></div>
        </div>
    `;
    
    html += `</div>`;
    
    resultsDiv.innerHTML = html;
}

// Rate answer
async function rateAnswer(rating) {
    if (!currentAnswer || !currentQuery) return;
    
    // Update star display
    const stars = document.querySelectorAll('.star');
    stars.forEach((star, index) => {
        if (index < rating) {
            star.classList.add('active');
        } else {
            star.classList.remove('active');
        }
    });
    
    try {
        const response = await fetch(`${API_BASE}/rate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: currentQuery,
                answer: currentAnswer,
                rating: rating
            })
        });
        
        if (!response.ok) {
            throw new Error('Rating failed');
        }
        
        document.getElementById('ratingFeedback').innerHTML = `
            <div style="color: #4caf50;">✅ Thank you for your feedback!</div>
        `;
        
    } catch (error) {
        document.getElementById('ratingFeedback').innerHTML = `
            <div style="color: #f44336;">❌ Failed to save rating</div>
        `;
    }
}

// Utility: Format file size
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}


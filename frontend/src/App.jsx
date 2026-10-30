import React, { useState, useEffect } from 'react'

function App() {
  const [activeTab, setActiveTab] = useState('search') // 'search', 'index', 'documents'
  const [apiUrl, setApiUrl] = useState('/api')

  // Health check & model states
  const [healthStatus, setHealthStatus] = useState(null)
  const [healthLoading, setHealthLoading] = useState(false)
  const [selectedModel, setSelectedModel] = useState('fine_tuned')
  const [modelSwitching, setModelSwitching] = useState(false)
  const [modelMessage, setModelMessage] = useState('')

  // Search tab state
  const [searchQuery, setSearchQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [searchResults, setSearchResults] = useState(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState('')

  // Index tab state (Document ID + Text only, no metadata)
  const [docId, setDocId] = useState('')
  const [docText, setDocText] = useState('')
  const [indexLoading, setIndexLoading] = useState(false)
  const [indexResult, setIndexResult] = useState(null)
  const [indexError, setIndexError] = useState('')

  // Documents view tab state
  const [documents, setDocuments] = useState([])
  const [docsLoading, setDocsLoading] = useState(false)
  const [docsError, setDocsError] = useState('')
  const [docFilter, setDocFilter] = useState('')

  // Fetch health check
  const checkHealth = async () => {
    setHealthLoading(true)
    try {
      const response = await fetch(`${apiUrl}/health`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setHealthStatus(data)
      if (data.model_id) {
        setSelectedModel(data.model_id)
      }
    } catch (err) {
      setHealthStatus({ status: 'offline', error: err.message })
    } finally {
      setHealthLoading(false)
    }
  }

  useEffect(() => {
    checkHealth()
  }, [apiUrl])

  // Fetch all documents
  const fetchDocuments = async () => {
    setDocsLoading(true)
    setDocsError('')
    try {
      const response = await fetch(`${apiUrl}/documents`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setDocuments(data.documents || [])
    } catch (err) {
      setDocsError(err.message)
    } finally {
      setDocsLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'documents') {
      fetchDocuments()
    }
  }, [activeTab, apiUrl])

  // Handle Model Switch
  const handleModelChange = async (newModelId) => {
    setSelectedModel(newModelId)
    setModelSwitching(true)
    setModelMessage('')
    try {
      const response = await fetch(`${apiUrl}/model/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: newModelId }),
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to switch model')
      }
      setModelMessage(`Active model switched to: ${data.active_model_name}`)
      checkHealth()
    } catch (err) {
      alert('Model switch error: ' + err.message)
    } finally {
      setModelSwitching(false)
    }
  }

  // Handle Search
  const handleSearch = async (e) => {
    e.preventDefault()
    if (!searchQuery.trim()) {
      alert('Please enter a search query')
      return
    }

    setSearchLoading(true)
    setSearchError('')
    setSearchResults(null)

    try {
      const response = await fetch(`${apiUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery.trim(),
          top_k: Number(topK),
          model_id: selectedModel,
        }),
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Search request failed')
      }
      setSearchResults(data)
    } catch (err) {
      setSearchError(err.message)
    } finally {
      setSearchLoading(false)
    }
  }

  // Handle Document Indexing (ID + Text only)
  const handleIndexDocument = async (e) => {
    e.preventDefault()
    if (!docId.trim() || !docText.trim()) {
      alert('Please fill out both ID and Text fields')
      return
    }

    setIndexLoading(true)
    setIndexError('')
    setIndexResult(null)

    const payload = {
      documents: [
        {
          id: docId.trim(),
          text: docText.trim(),
        },
      ],
    }

    try {
      const response = await fetch(`${apiUrl}/index`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Index request failed')
      }
      setIndexResult(data)
      setDocId('')
      setDocText('')
      checkHealth()
      if (activeTab === 'documents') {
        fetchDocuments()
      }
    } catch (err) {
      setIndexError(err.message)
    } finally {
      setIndexLoading(false)
    }
  }

  // Filtered documents
  const filteredDocs = documents.filter((doc) => {
    if (!docFilter.trim()) return true
    const q = docFilter.toLowerCase()
    return doc.id.toLowerCase().includes(q) || doc.text.toLowerCase().includes(q)
  })

  return (
    <div className="container">
      <h1>Semantic Search Engine</h1>
      <p style={{ fontSize: '13px', color: '#555555', marginTop: '4px' }}>
        Dense-vector retrieval powered by SentenceTransformers &amp; FAISS. Search across technical documentation and Python code implementations.
      </p>

      {/* Backend Connection & Model Selector Box */}
      <div className="status-box">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div>
            <strong>Backend API: </strong>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              style={{ width: '180px', padding: '4px 6px', fontSize: '13px' }}
            />
            <button
              onClick={checkHealth}
              disabled={healthLoading}
              style={{ marginLeft: '8px', padding: '4px 10px', fontSize: '12px', cursor: 'pointer' }}
            >
              {healthLoading ? 'Checking...' : 'Check Status'}
            </button>
          </div>
          <div>
            Status:{' '}
            <span
              className={`status-indicator ${healthStatus && healthStatus.status === 'healthy' ? 'status-ok' : 'status-bad'
                }`}
            >
              {healthStatus ? healthStatus.status.toUpperCase() : 'UNKNOWN'}
            </span>
            {healthStatus && healthStatus.status === 'healthy' && (
              <span style={{ marginLeft: '10px', color: '#555' }}>
                (<b>{healthStatus.index_size}</b> vectors indexed)
              </span>
            )}
          </div>
        </div>

        {/* Model Selector Feature */}
        <div className="model-selector-row">
          <label style={{ margin: 0, whiteSpace: 'nowrap' }}>
            🧠 <strong>Choose Model:</strong>
          </label>
          <select
            className="model-select"
            value={selectedModel}
            disabled={modelSwitching}
            onChange={(e) => handleModelChange(e.target.value)}
          >
            <option value="fine_tuned">
              🚀 Fine-Tuned Model (Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search)
            </option>
            <option value="base">
              📦 Base Model (sentence-transformers/all-MiniLM-L6-v2)
            </option>
          </select>
          {modelSwitching && <span style={{ fontSize: '12px', color: '#666' }}>Switching model...</span>}
        </div>

        {modelMessage && (
          <div style={{ fontSize: '12px', color: '#155724', marginTop: '6px' }}>
            ✓ {modelMessage}
          </div>
        )}
        {healthStatus && healthStatus.status !== 'healthy' && (
          <div style={{ fontSize: '12px', color: '#c92a2a', marginTop: '6px' }}>
            Could not connect to FastAPI server at {apiUrl}. Make sure your backend is running.
          </div>
        )}
      </div>

      {/* Navigation Tabs */}
      <div className="nav-tabs">
        <button
          className={`tab-btn ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          🔍 1. Search
        </button>
        <button
          className={`tab-btn ${activeTab === 'index' ? 'active' : ''}`}
          onClick={() => setActiveTab('index')}
        >
          ➕ 2. Add Document
        </button>
        <button
          className={`tab-btn ${activeTab === 'documents' ? 'active' : ''}`}
          onClick={() => setActiveTab('documents')}
        >
          📄 3. View Documents {documents.length > 0 ? `(${documents.length})` : ''}
        </button>
      </div>

      {/* TAB 1: SEARCH */}
      {activeTab === 'search' && (
        <div>
          <h2>Search Documents &amp; Code</h2>
          <form onSubmit={handleSearch}>
            <div className="form-group">
              <label>Enter Query (Natural Language or Code Search):</label>
              <input
                type="text"
                placeholder="e.g. binary search in sorted array, or how do transformers process sequential data?"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            <div className="form-group" style={{ width: '150px' }}>
              <label>Top Results (k):</label>
              <input
                type="number"
                min="1"
                max="50"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
              />
            </div>

            <button type="submit" className="submit-btn" disabled={searchLoading}>
              {searchLoading ? 'Searching...' : 'Search'}
            </button>
          </form>

          {searchError && (
            <div className="error-message">
              <b>Error:</b> {searchError}
            </div>
          )}

          {searchResults && (
            <div style={{ marginTop: '22px' }}>
              <h2>Search Results</h2>
              <div className="results-meta">
                Found <b>{searchResults.results.length}</b> result(s) in <b>{searchResults.latency_ms} ms</b> using model:{' '}
                <code>{searchResults.model_name}</code>
              </div>

              {searchResults.results.length === 0 ? (
                <p>No matching results found.</p>
              ) : (
                searchResults.results.map((item) => {
                  const isCode = item.document_id.startsWith('code-') || item.text.includes('def ') || item.text.includes('class ')
                  return (
                    <div key={item.document_id + item.rank} className="result-card">
                      <div className="result-header">
                        <div>
                          <span className={`badge ${isCode ? 'badge-code' : 'badge-doc'}`}>
                            {isCode ? 'CODE' : 'DOC'}
                          </span>
                          <span className="result-rank">
                            #{item.rank} &bull; {item.document_id}
                          </span>
                        </div>
                        <span className="result-score">
                          Cosine Similarity: {(item.score * 100).toFixed(1)}% ({item.score.toFixed(4)})
                        </span>
                      </div>
                      <div className={isCode ? 'code-preview' : 'result-text'}>
                        {item.text}
                      </div>
                    </div>
                  )
                })
              )}
            </div>
          )}
        </div>
      )}

      {/* TAB 2: INDEX (No metadata asked from UI) */}
      {activeTab === 'index' && (
        <div>
          <h2>Add / Index New Document</h2>
          <p style={{ fontSize: '13px', color: '#666', marginTop: 0 }}>
            Embed and index new text or code into the FAISS vector space.
          </p>
          <form onSubmit={handleIndexDocument}>
            <div className="form-group">
              <label>Document ID:</label>
              <input
                type="text"
                placeholder="e.g. doc-101 or code-021"
                value={docId}
                onChange={(e) => setDocId(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label>Document Text / Code:</label>
              <textarea
                rows="6"
                placeholder="Enter text or Python code implementation to embed into vector space..."
                value={docText}
                onChange={(e) => setDocText(e.target.value)}
              />
            </div>

            <button type="submit" className="submit-btn" disabled={indexLoading}>
              {indexLoading ? 'Indexing...' : 'Index Document'}
            </button>
          </form>

          {indexError && (
            <div className="error-message">
              <b>Error:</b> {indexError}
            </div>
          )}

          {indexResult && (
            <div className="success-message">
              <b>Success:</b> Indexed {indexResult.indexed} document. Total vectors in index is now{' '}
              {indexResult.total_index_size}.
            </div>
          )}
        </div>
      )}

      {/* TAB 3: VIEW DOCUMENTS */}
      {activeTab === 'documents' && (
        <div>
          <h2>Corpus Documents</h2>
          <p style={{ fontSize: '13px', color: '#666', marginTop: 0 }}>
            Browse all documents and code implementations currently stored in the search index.
          </p>

          <div className="docs-toolbar">
            <input
              type="text"
              placeholder="Filter by ID or content (e.g. 'code', 'docker', 'sort')..."
              value={docFilter}
              onChange={(e) => setDocFilter(e.target.value)}
            />
            <button
              onClick={fetchDocuments}
              disabled={docsLoading}
              style={{ padding: '7px 14px', fontSize: '13px', cursor: 'pointer' }}
            >
              {docsLoading ? 'Refreshing...' : '🔄 Refresh'}
            </button>
          </div>

          {docsError && (
            <div className="error-message">
              <b>Error:</b> {docsError}
            </div>
          )}

          <div style={{ fontSize: '13px', color: '#666', marginBottom: '10px' }}>
            Showing <b>{filteredDocs.length}</b> of <b>{documents.length}</b> total documents
          </div>

          {filteredDocs.length === 0 && !docsLoading && (
            <p style={{ color: '#888' }}>No documents matched your filter.</p>
          )}

          <div>
            {filteredDocs.map((doc, idx) => {
              const isCode = doc.id.startsWith('code-') || doc.text.includes('def ') || doc.text.includes('class ')
              return (
                <div key={doc.id} className="doc-item-card">
                  <div className="doc-item-header">
                    <div>
                      <span className={`badge ${isCode ? 'badge-code' : 'badge-doc'}`}>
                        {isCode ? 'CODE' : 'DOC'}
                      </span>
                      <span className="doc-id-label">{doc.id}</span>
                    </div>
                    <span style={{ fontSize: '11px', color: '#888' }}>
                      #{idx + 1} &bull; {doc.text.length} chars
                    </span>
                  </div>
                  <div className={isCode ? 'code-preview' : 'result-text'} style={{ fontSize: '13px' }}>
                    {doc.text}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      <hr style={{ marginTop: '30px', borderColor: '#eeeeee' }} />
      <div style={{ textAlign: 'center', fontSize: '11px', color: '#888888' }}>
        Semantic Search Engine &bull; Serving on Port 9092
      </div>
    </div>
  )
}

export default App

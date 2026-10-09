import React, { useState, useEffect } from 'react'

function App() {
  const [activeTab, setActiveTab] = useState('search') // 'search' or 'index'
  const [apiUrl, setApiUrl] = useState('/api')

  // Health check state
  const [healthStatus, setHealthStatus] = useState(null)
  const [healthLoading, setHealthLoading] = useState(false)

  // Search tab state
  const [searchQuery, setSearchQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [searchResults, setSearchResults] = useState(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState('')

  // Index tab state
  const [docId, setDocId] = useState('')
  const [docText, setDocText] = useState('')
  const [docCategory, setDocCategory] = useState('')
  const [indexLoading, setIndexLoading] = useState(false)
  const [indexResult, setIndexResult] = useState(null)
  const [indexError, setIndexError] = useState('')

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
    } catch (err) {
      setHealthStatus({ status: 'offline', error: err.message })
    } finally {
      setHealthLoading(false)
    }
  }

  useEffect(() => {
    checkHealth()
  }, [apiUrl])

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
          query: searchQuery,
          top_k: Number(topK),
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

  // Handle Document Indexing
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
          metadata: docCategory.trim() ? { category: docCategory.trim() } : {},
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
      setDocCategory('')
      // Refresh health to get updated index_size
      checkHealth()
    } catch (err) {
      setIndexError(err.message)
    } finally {
      setIndexLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>Semantic Search Engine</h1>
      <p style={{ fontSize: '13px', color: '#555555' }}>
        A simple React frontend to query and add documents to your FastAPI + FAISS semantic search engine.
      </p>

      {/* Backend Connection & Status Box */}
      <div className="status-box">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div>
            <strong>Backend API: </strong>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              style={{ width: '220px', padding: '4px', fontSize: '13px' }}
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
              className={`status-indicator ${
                healthStatus && healthStatus.status === 'healthy' ? 'status-ok' : 'status-bad'
              }`}
            >
              {healthStatus ? healthStatus.status.toUpperCase() : 'UNKNOWN'}
            </span>
          </div>
        </div>

        {healthStatus && healthStatus.status === 'healthy' && (
          <div style={{ fontSize: '12px', color: '#444444' }}>
            Loaded Model: <code>{healthStatus.model_name}</code> | Total Vectors Indexed: <b>{healthStatus.index_size}</b>
          </div>
        )}
        {healthStatus && healthStatus.status !== 'healthy' && (
          <div style={{ fontSize: '12px', color: 'red' }}>
            Could not connect to FastAPI server at {apiUrl}. Make sure your backend is running.
          </div>
        )}
      </div>

      {/* Simple Navigation Tabs */}
      <div className="nav-tabs">
        <button
          className={`tab-btn ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          1. Search Documents
        </button>
        <button
          className={`tab-btn ${activeTab === 'index' ? 'active' : ''}`}
          onClick={() => setActiveTab('index')}
        >
          2. Add Document
        </button>
      </div>

      {/* TAB 1: SEARCH */}
      {activeTab === 'search' && (
        <div>
          <h2>Search Queries</h2>
          <form onSubmit={handleSearch}>
            <div className="form-group">
              <label>Enter Query:</label>
              <input
                type="text"
                placeholder="e.g. How does Docker containerization work?"
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
            <div style={{ marginTop: '20px' }}>
              <h2>Search Results</h2>
              <div className="results-meta">
                Found {searchResults.results.length} result(s) in {searchResults.latency_ms} ms for query: "
                <b>{searchResults.query}</b>"
              </div>

              {searchResults.results.length === 0 ? (
                <p>No matching results found.</p>
              ) : (
                searchResults.results.map((item) => (
                  <div key={item.document_id + item.rank} className="result-card">
                    <div className="result-header">
                      <span className="result-rank">
                        #{item.rank} - ID: {item.document_id}
                      </span>
                      <span className="result-score">
                        Cosine Similarity: {(item.score * 100).toFixed(1)}% ({item.score.toFixed(4)})
                      </span>
                    </div>
                    <div className="result-text">{item.text}</div>
                    {item.metadata && Object.keys(item.metadata).length > 0 && (
                      <div className="result-meta">
                        <strong>Metadata:</strong> {JSON.stringify(item.metadata)}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      )}

      {/* TAB 2: INDEX */}
      {activeTab === 'index' && (
        <div>
          <h2>Add / Index New Document</h2>
          <form onSubmit={handleIndexDocument}>
            <div className="form-group">
              <label>Document ID:</label>
              <input
                type="text"
                placeholder="e.g. doc-101"
                value={docId}
                onChange={(e) => setDocId(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label>Document Text:</label>
              <textarea
                rows="4"
                placeholder="Enter text content to embed into vector space..."
                value={docText}
                onChange={(e) => setDocText(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label>Category (Optional Metadata):</label>
              <input
                type="text"
                placeholder="e.g. devops, machine-learning, database"
                value={docCategory}
                onChange={(e) => setDocCategory(e.target.value)}
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

      <hr style={{ marginTop: '30px', borderColor: '#eeeeee' }} />
      <div style={{ textAlign: 'center', fontSize: '11px', color: '#888888' }}>
        Semantic Search UI &bull; Running on Port 9092
      </div>
    </div>
  )
}

export default App

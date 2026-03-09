import { useState, useEffect, useMemo } from 'react'
import { Search, FileText, AlertCircle } from 'lucide-react'

export default function Articles() {
    const [articles, setArticles] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [search, setSearch] = useState('')

    useEffect(() => {
        fetch('/api/articles')
            .then((res) => {
                if (!res.ok) throw new Error(`Server returned ${res.status}`)
                return res.json()
            })
            .then((data) => {
                setArticles(data.articles || [])
            })
            .catch((err) => {
                setError(err.message || 'Failed to load articles.')
            })
            .finally(() => setLoading(false))
    }, [])

    const filtered = useMemo(() => {
        if (!search.trim()) return articles
        const q = search.toLowerCase()
        return articles.filter(
            (a) =>
                (a.article_no || '').toLowerCase().includes(q) ||
                (a.title || '').toLowerCase().includes(q)
        )
    }, [articles, search])

    return (
        <div className="articles-page">
            {/* ── Header ───────────────────────────────────── */}
            <div className="articles-header">
                <h1>Browse the Constitution</h1>
                <p>All articles from the Constitution of India — search by number or title.</p>
            </div>

            {/* ── Toolbar ──────────────────────────────────── */}
            <div className="articles-toolbar">
                <div className="articles-search-box">
                    <Search size={18} />
                    <input
                        id="articles-search"
                        type="text"
                        placeholder="Search articles by number or title..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                    />
                </div>
                {!loading && !error && (
                    <div className="articles-count-badge">
                        Showing <strong>{filtered.length}</strong> of {articles.length} articles
                    </div>
                )}
            </div>

            {/* ── Loading Skeletons ────────────────────────── */}
            {loading && (
                <div className="articles-list">
                    {Array.from({ length: 8 }).map((_, i) => (
                        <div key={i} className="skeleton-row" style={{ animationDelay: `${i * 0.05}s` }} />
                    ))}
                </div>
            )}

            {/* ── Error ────────────────────────────────────── */}
            {error && !loading && (
                <div className="response-section" style={{ maxWidth: 760, margin: '0 auto' }}>
                    <div className="error-card">
                        <AlertCircle size={20} />
                        <div className="error-card-content">
                            <h4>Failed to load articles</h4>
                            <p>{error}</p>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Articles List ────────────────────────────── */}
            {!loading && !error && (
                <>
                    {filtered.length === 0 ? (
                        <div className="empty-state">
                            <FileText size={40} />
                            <h3>No articles found</h3>
                            <p>Try a different search term.</p>
                        </div>
                    ) : (
                        <div className="articles-list">
                            {filtered.map((article, i) => (
                                <div
                                    key={article.article_no || i}
                                    className="article-row animate-fade-in-up"
                                    style={{ animationDelay: `${Math.min(i * 0.03, 0.5)}s`, opacity: 0 }}
                                >
                                    <div className="article-number">
                                        Art. {article.article_no}
                                    </div>
                                    <div className="article-info">
                                        <div className="article-title">{article.title || 'Untitled'}</div>
                                    </div>
                                    <span
                                        className={`article-status ${(article.status || 'Active').toLowerCase() === 'omitted'
                                                ? 'omitted'
                                                : 'active'
                                            }`}
                                    >
                                        {article.status || 'Active'}
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </>
            )}
        </div>
    )
}

import { useState } from 'react'
import { Send, Search, BookOpen, Shield, Cpu, AlertCircle, Scale } from 'lucide-react'

const SUGGESTIONS = [
    'What does Article 14 say about equality?',
    'Right to freedom of speech',
    'What is Article 21?',
    'Rights against exploitation',
]

const FEATURES = [
    {
        icon: Search,
        title: 'Semantic Search',
        desc: 'Find relevant constitutional articles using natural language queries, powered by vector similarity.',
    },
    {
        icon: Shield,
        title: 'Legally Grounded',
        desc: 'Every answer is strictly grounded in the text of the Constitution — no hallucinations.',
    },
    {
        icon: Cpu,
        title: 'Local & Private',
        desc: 'Runs entirely on your machine using Ollama. Your queries never leave your computer.',
    },
    {
        icon: BookOpen,
        title: 'Source Citations',
        desc: 'Each response includes the exact Article numbers and Parts used to generate the answer.',
    },
]

export default function Home() {
    const [query, setQuery] = useState('')
    const [loading, setLoading] = useState(false)
    const [response, setResponse] = useState(null)
    const [error, setError] = useState(null)

    const handleSubmit = async (e) => {
        e?.preventDefault()
        const q = query.trim()
        if (!q || loading) return

        setLoading(true)
        setError(null)
        setResponse(null)

        try {
            const res = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: q }),
            })

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}))
                throw new Error(errData.detail || `Server returned ${res.status}`)
            }

            const data = await res.json()
            setResponse(data)
        } catch (err) {
            setError(err.message || 'Something went wrong. Is the backend running?')
        } finally {
            setLoading(false)
        }
    }

    const handleSuggestion = (text) => {
        setQuery(text)
        setTimeout(() => {
            setLoading(true)
            setError(null)
            setResponse(null)
            fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text }),
            })
                .then((res) => {
                    if (!res.ok) throw new Error(`Server returned ${res.status}`)
                    return res.json()
                })
                .then((data) => setResponse(data))
                .catch((err) => setError(err.message))
                .finally(() => setLoading(false))
        }, 100)
    }

    return (
        <>
            {/* ── Hero ─────────────────────────────────────────── */}
            <section className="hero">
                <div className="hero-content">
                    <div className="hero-badge">
                        <Scale size={14} /> AI-Powered Legal Assistant
                    </div>
                    <h1>
                        Your Guide to the{' '}
                        <span className="highlight">Indian Constitution</span>
                    </h1>
                    <p className="hero-subtitle">
                        Ask any legal question about the Constitution of India and get
                        accurate, article-referenced answers powered by AI.
                    </p>
                </div>
            </section>

            {/* ── Query Box ────────────────────────────────────── */}
            <section className="query-section">
                <form onSubmit={handleSubmit}>
                    <div className="query-box">
                        <div className="query-input-row">
                            <Search size={18} color="var(--text-tertiary)" style={{ marginLeft: 12, flexShrink: 0 }} />
                            <input
                                id="query-input"
                                type="text"
                                placeholder="Ask a question about the Indian Constitution..."
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                autoComplete="off"
                            />
                            <button
                                id="query-submit"
                                type="submit"
                                className="query-submit-btn"
                                disabled={loading || query.trim().length < 3}
                            >
                                {loading ? (
                                    <>Thinking...</>
                                ) : (
                                    <>
                                        <Send size={16} /> Ask Harvey
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </form>

                <div className="query-suggestions">
                    {SUGGESTIONS.map((s) => (
                        <button
                            key={s}
                            className="query-suggestion"
                            onClick={() => handleSuggestion(s)}
                            type="button"
                        >
                            {s}
                        </button>
                    ))}
                </div>
            </section>

            {/* ── Loading ──────────────────────────────────────── */}
            {loading && (
                <div className="response-section">
                    <div className="response-card">
                        <div className="typing-indicator">
                            <div className="typing-dots">
                                <div className="typing-dot" />
                                <div className="typing-dot" />
                                <div className="typing-dot" />
                            </div>
                            <span className="typing-text">Harvey is analyzing the Constitution...</span>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Error ────────────────────────────────────────── */}
            {error && !loading && (
                <div className="response-section">
                    <div className="error-card">
                        <AlertCircle size={20} />
                        <div className="error-card-content">
                            <h4>Unable to get a response</h4>
                            <p>{error}</p>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Response ─────────────────────────────────────── */}
            {response && !loading && (
                <div className="response-section">
                    <div className="response-card">
                        <div className="response-header">
                            <div className="response-header-left">
                                <div className="response-avatar">
                                    <Scale size={16} color="#0a0e17" />
                                </div>
                                <span className="response-header-label">Harvey Spector</span>
                            </div>
                        </div>
                        <div className="response-body">
                            <div className="response-answer">{response.answer}</div>
                            {response.sources && response.sources.length > 0 && (
                                <div className="response-sources">
                                    <div className="response-sources-title">Referenced Articles</div>
                                    <div className="source-chips">
                                        {response.sources.map((src, i) => (
                                            <span key={i} className="source-chip">
                                                <BookOpen size={14} />
                                                Art. {src.article_no} — {src.title}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* ── Features ─────────────────────────────────────── */}
            {!response && !loading && !error && (
                <section className="features-section">
                    <div className="features-grid">
                        {FEATURES.map(({ icon: Icon, title, desc }, i) => (
                            <div
                                key={title}
                                className="feature-card"
                                style={{ animationDelay: `${i * 0.1}s` }}
                            >
                                <div className="feature-icon">
                                    <Icon size={22} />
                                </div>
                                <h3>{title}</h3>
                                <p>{desc}</p>
                            </div>
                        ))}
                    </div>
                </section>
            )}
        </>
    )
}

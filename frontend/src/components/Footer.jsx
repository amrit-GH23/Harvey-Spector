import { Scale, Github } from 'lucide-react'

export default function Footer() {
    return (
        <footer className="footer">
            <div className="footer-inner">
                <div className="footer-text">
                    <Scale size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />
                    Built with <span>Harvey Spector</span> — Powered by Ollama + ChromaDB + LangChain
                </div>
                <div className="footer-links">
                    <a href="/docs" target="_blank" rel="noreferrer">API Docs</a>
                    <a
                        href="https://github.com"
                        target="_blank"
                        rel="noreferrer"
                        style={{ display: 'flex', alignItems: 'center', gap: 4 }}
                    >
                        <Github size={14} /> GitHub
                    </a>
                </div>
            </div>
        </footer>
    )
}

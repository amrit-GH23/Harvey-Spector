import { useState } from 'react'
import { NavLink, Link } from 'react-router-dom'
import { Scale, Menu, X } from 'lucide-react'

export default function Navbar() {
    const [mobileOpen, setMobileOpen] = useState(false)

    return (
        <nav className="navbar">
            <div className="navbar-inner">
                <Link to="/" className="navbar-brand">
                    <div className="navbar-brand-icon">
                        <Scale size={18} color="#0a0e17" />
                    </div>
                    <div className="navbar-brand-text">
                        Harvey <span>Spector</span>
                    </div>
                </Link>

                <ul className={`navbar-links ${mobileOpen ? 'open' : ''}`}>
                    <li>
                        <NavLink
                            to="/"
                            end
                            className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
                            onClick={() => setMobileOpen(false)}
                        >
                            Ask Harvey
                        </NavLink>
                    </li>
                    <li>
                        <NavLink
                            to="/articles"
                            className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
                            onClick={() => setMobileOpen(false)}
                        >
                            Browse Articles
                        </NavLink>
                    </li>
                </ul>

                <button
                    className="navbar-mobile-toggle"
                    onClick={() => setMobileOpen(!mobileOpen)}
                    aria-label="Toggle menu"
                >
                    {mobileOpen ? <X size={22} /> : <Menu size={22} />}
                </button>
            </div>
        </nav>
    )
}

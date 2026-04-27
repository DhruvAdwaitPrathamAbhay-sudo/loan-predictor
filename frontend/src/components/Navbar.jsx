import { Link, useLocation } from 'react-router-dom';
import { Home, FileText, PieChart } from 'lucide-react';
import './Navbar.css';

const Navbar = () => {
  const location = useLocation();

  const navLinks = [
    { path: '/', label: 'Home', icon: <Home size={20} /> },
    { path: '/apply', label: 'Apply Now', icon: <FileText size={20} /> },
    { path: '/result', label: 'Results', icon: <PieChart size={20} /> },
  ];

  return (
    <nav className="glass-navbar">
      <div className="navbar-brand">
        <div className="status-dot"></div>
        <h2>Smart Finance</h2>
      </div>
      <div className="navbar-links">
        {navLinks.map((link) => (
          <Link
            key={link.path}
            to={link.path}
            className={`nav-link ${location.pathname === link.path ? 'active' : ''}`}
          >
            {link.icon}
            <span>{link.label}</span>
          </Link>
        ))}
      </div>
    </nav>
  );
};

export default Navbar;

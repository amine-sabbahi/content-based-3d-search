import { useState } from 'react';

const RegisterForm = () => {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [message, setMessage] = useState('');

  const handleRegister = async () => {
    if (!fullName || !email || !password || !confirmPassword) {
      setMessage('All fields are required');
      return;
    }
    if (password !== confirmPassword) {
      setMessage('Passwords do not match');
      return;
    }

    try {
      const response = await fetch('http://127.0.0.1:5000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fullName, email, password }),
      });
      const data = await response.json();
      setMessage(data.message);
    } catch (error) {
      console.error('Register Error:', error);
      setMessage('An error occurred while registering');
    }
  };

  return (
    <div className="register-container">
      <h2>Register</h2>
      <div className="form-group">
        <label>Full Name</label>
        <input
          type="text"
          value={fullName}
          onChange={(e) => setFullName(e.target.value)}
          required
        />
      </div>
      <div className="form-group">
        <label>Email</label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
      </div>
      <div className="form-group">
        <label>Password</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </div>
      <div className="form-group">
        <label>Confirm Password</label>
        <input
          type="password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required
        />
      </div>
      <button onClick={handleRegister}>Register</button>
      <p className="message">{message}</p>
      <style jsx>{`
        .register-container {
          max-width: 400px;
          margin: auto;
          padding: 20px;
          border-radius: 10px;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          background-color: #f9f9f9;
        }
        h2 {
          text-align: center;
          color: #333;
        }
        .form-group {
          margin-bottom: 15px;
        }
        label {
          display: block;
          margin-bottom: 5px;
          color: #555;
        }
        input {
          width: 100%;
          padding: 10px;
          border: 1px solid #ccc;
          border-radius: 5px;
        }
        button {
          width: 100%;
          padding: 10px;
          background-color: #0070f3;
          color: white;
          border: none;
          border-radius: 5px;
          cursor: pointer;
          font-size: 16px;
        }
        button:hover {
          background-color: #005bb5;
        }
        .message {
          margin-top: 15px;
          color: red;
          text-align: center;
        }
      `}</style>
    </div>
  );
};

export default RegisterForm;

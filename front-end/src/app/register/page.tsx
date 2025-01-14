"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Eye, EyeOff } from 'lucide-react';

const RegisterPage = () => {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [message, setMessage] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const router = useRouter();

  // Validation functions
  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const validatePassword = (password) => {
    // Password requirements:
    // - At least 8 characters
    // - Contains at least one uppercase letter
    // - Contains at least one lowercase letter
    // - Contains at least one number
    // - Contains at least one special character
    const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
    return passwordRegex.test(password);
  };

  const handleRegister = async (e) => {
    e.preventDefault(); // Prevent default form submission

    // Reset previous message
    setMessage('');

    // Comprehensive validation
    if (!fullName.trim()) {
      setMessage('Full name is required');
      return;
    }

    if (!validateEmail(email)) {
      setMessage('Please enter a valid email address');
      return;
    }

    if (!validatePassword(password)) {
      setMessage('Password must be at least 8 characters long and include uppercase, lowercase, number, and special character');
      return;
    }

    if (password !== confirmPassword) {
      setMessage('Passwords do not match');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/register', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-CSRF-Token': await getCsrfToken() // Add CSRF protection
        },
        body: JSON.stringify({ 
          fullName: fullName.trim(), 
          email: email.trim().toLowerCase(), 
          password 
        }),
        credentials: 'same-origin' // Important for security
      });

      const data = await response.json();

      if (response.ok) {
        // Clear sensitive information
        setFullName('');
        setEmail('');
        setPassword('');
        setConfirmPassword('');

        // Show success message before redirecting
        setMessage('Registration successful. Redirecting...');
        
        // Short delay before redirecting to allow user to see success message
        setTimeout(() => {
          router.push('/login');
        }, 1500);
      } else {
        // Handle specific error messages from backend
        setMessage(data.message || 'Registration failed');
      }
    } catch (error) {
      console.error('Register Error:', error);
      setMessage('An unexpected error occurred. Please try again.');
    }
  };

  // Simulated CSRF token retrieval (replace with your actual implementation)
  const getCsrfToken = async () => {
    // In a real app, this would be fetched from your backend
    return 'simulated-csrf-token';
  };

  // Toggle password visibility
  const togglePasswordVisibility = (field) => {
    if (field === 'password') {
      setShowPassword(!showPassword);
    } else {
      setShowConfirmPassword(!showConfirmPassword);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 p-4">
      <div className="bg-white shadow-md rounded-lg p-8 max-w-md w-full">
        <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">Create Your Account</h2>
        
        <form onSubmit={handleRegister} className="space-y-4">
          <div>
            <label htmlFor="fullName" className="block text-sm font-medium text-gray-700 mb-1">
              Full Name
            </label>
            <input
              id="fullName"
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter your full name"
              required
              aria-required="true"
            />
          </div>

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
              Email Address
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="you@example.com"
              required
              aria-required="true"
            />
          </div>

          <div className="relative">
            <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <input
              id="password"
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 pr-10"
              placeholder="Create a strong password"
              required
              aria-required="true"
            />
            <button
              type="button"
              onClick={() => togglePasswordVisibility('password')}
              className="absolute inset-y-0 right-0 px-3 flex items-center text-gray-500 focus:outline-none"
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>

          <div className="relative">
            <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700 mb-1">
              Confirm Password
            </label>
            <input
              id="confirmPassword"
              type={showConfirmPassword ? "text" : "password"}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 pr-10"
              placeholder="Confirm your password"
              required
              aria-required="true"
            />
            <button
              type="button"
              onClick={() => togglePasswordVisibility('confirmPassword')}
              className="absolute inset-y-0 right-0 px-3 flex items-center text-gray-500 focus:outline-none"
              aria-label={showConfirmPassword ? "Hide password" : "Show password"}
            >
              {showConfirmPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>

          <button
            type="submit"
            className="w-full bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600 transition duration-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
          >
            Register
          </button>

          {message && (
            <p 
              className={`mt-4 text-center ${
                message.includes('successful') 
                  ? 'text-green-600' 
                  : 'text-red-600'
              }`}
              role="alert"
            >
              {message}
            </p>
          )}
        </form>
      </div>
    </div>
  );
};

export default RegisterPage;
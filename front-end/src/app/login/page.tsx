//app/login/pages.tsx

"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Eye, EyeOff } from 'lucide-react';

const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const router = useRouter();

  // Email validation function
  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleLogin = async (e) => {
    e.preventDefault(); // Prevent default form submission

    // Reset previous message
    setMessage('');

    // Validate email
    if (!validateEmail(email)) {
      setMessage('Please enter a valid email address');
      return;
    }

    // Check if password is not empty
    if (!password.trim()) {
      setMessage('Please enter your password');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/login', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-CSRF-Token': await getCsrfToken() // Add CSRF protection
        },
        body: JSON.stringify({ 
          email: email.trim().toLowerCase(), 
          password 
        }),
        credentials: 'same-origin' // Important for security
      });

      const data = await response.json();

      if (response.ok) {
        // Clear sensitive information
        setEmail('');
        setPassword('');

        // Show success message before redirecting
        setMessage('Login successful. Redirecting...');
        
        // Short delay before redirecting to allow user to see success message
        setTimeout(() => {
          router.push('/dashboard');
        }, 1500);
      } else {
        // Handle specific error messages from backend
        setMessage(data.message || 'Login failed');
      }
    } catch (error) {
      console.error('Login Error:', error);
      setMessage('An unexpected error occurred. Please try again.');
    }
  };

  // Simulated CSRF token retrieval (replace with your actual implementation)
  const getCsrfToken = async () => {
    // In a real app, this would be fetched from your backend
    return 'simulated-csrf-token';
  };

  // Toggle password visibility
  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 p-4">
      <div className="bg-white shadow-md rounded-lg p-8 max-w-md w-full">
        <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">Welcome Back</h2>
        
        <form onSubmit={handleLogin} className="space-y-4">
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
              placeholder="Enter your password"
              required
              aria-required="true"
            />
            <button
              type="button"
              onClick={togglePasswordVisibility}
              className="absolute inset-y-0 right-0 px-3 flex items-center text-gray-500 focus:outline-none"
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>

          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <input
                id="remember-me"
                type="checkbox"
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label 
                htmlFor="remember-me" 
                className="ml-2 block text-sm text-gray-900"
              >
                Remember me
              </label>
            </div>
            <div className="text-sm">
              <a 
                href="/forgot-password" 
                className="font-medium text-blue-600 hover:text-blue-500"
              >
                Forgot password?
              </a>
            </div>
          </div>

          <button
            type="submit"
            className="w-full bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600 transition duration-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
          >
            Login
          </button>

          <div className="text-center mt-4">
            <p className="text-sm text-gray-600">
              Don't have an account?{' '}
              <a 
                href="/register" 
                className="font-medium text-blue-600 hover:text-blue-500"
              >
                Register here
              </a>
            </p>
          </div>

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

export default LoginPage;
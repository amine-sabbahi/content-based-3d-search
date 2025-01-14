// components/SideBar.tsx
'use client';
import React, {  useState } from 'react';
import { 
  Upload, 
  ImagePlus, 
  Repeat2, 
  MessageCircleQuestion, 
  ChevronLeft, 
  ChevronRight 
} from 'lucide-react';
import { usePathname, useRouter } from 'next/navigation';

const SideBarImageApp = ({children}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [removedItems] = useState([]);
  const pathname = usePathname();
  const router = useRouter();

  const toggleSidebar = () => setIsCollapsed(!isCollapsed);

  // Function to check if the path is active
  const isActive = (paths) => paths.includes(pathname);

  const handleLogout = () => {
    // Perform any logout logic here if necessary (e.g., clearing tokens, etc.)
    router.push('/'); // Redirect to the home page
  };
  // Navigation items with icons and labels
  const navigationItems = [
    {icon: <Upload className="w-5 h-5"/>, label: 'Upload Images', href: '/dashboard'},
    {icon: <ImagePlus className="w-5 h-5"/>, label: 'Transform Images', href: '/dashboard/transform-images'},
    {icon: <Repeat2 className="w-5 h-5"/>, label: 'Simple Return', href: '/dashboard/simple-return'},
    {icon: <Repeat2 className="w-5 h-5"/>, label: 'Feedback', href: '/dashboard/feedback-return'},
  ];

  return (
    <div className="flex flex-row h-screen">
      {/* Sidebar */}
      <div
        className={`
          fixed left-0 top-0 h-full 
          bg-gradient-to-b from-[#2c3e50] to-[#34495e] 
          text-white shadow-2xl 
          transition-all duration-300 
          ${isCollapsed ? 'w-16' : 'w-64'}
          flex flex-col 
          z-50 
          border-r border-gray-700
        `}
      >
        {/* Collapse/Expand Button */}
        <button
          onClick={toggleSidebar}
          className="
            absolute top-4 right-[-20px] 
            bg-[#34495e] 
            text-white 
            p-2 
            rounded-r-full 
            shadow-lg 
            hover:bg-[#2c3e50] 
            transition-colors 
            group
          "
        >
          {isCollapsed ? (
            <ChevronRight className="w-5 h-5 group-hover:animate-pulse"/>
          ) : (
            <ChevronLeft className="w-5 h-5 group-hover:animate-pulse"/>
          )}
        </button>

        {}
        
        <div className="p-4 border-b border-gray-700 flex items-center justify-center bg">
          <div 
            className={`
              transition-all 
              ${isCollapsed ? 'w-10 h-10 overflow-hidden' : 'w-full'}
              flex justify-center items-center
            `}
          >
            {'Dashboard'}
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="flex-grow mt-4 space-y-1">
          {navigationItems
            .filter((item) => !removedItems.includes(item.href))
            .map((item, index) => (
              <div key={index} className="relative group">
                <a
                  href={item.href}
                  className={`
                    flex items-center 
                    px-4 py-3 
                    transition-all 
                    duration-300 
                    ease-in-out 
                    ${
                      isActive([item.href])
                        ? 'bg-[#3498db] text-white scale-105'
                        : 'hover:bg-[#34495e] text-gray-300 hover:scale-105'
                    }
                    transform 
                    hover:shadow-lg 
                    rounded-r-xl 
                    mx-2 
                    group
                  `}
                >
                  <span className={`
                    ${isActive([item.href]) ? 'text-white' : 'text-gray-400'}
                    group-hover:text-white
                    transition-colors
                  `}>
                    {item.icon}
                  </span>
                  <span
                    className={`
                      ml-3 
                      transition-all 
                      ${isCollapsed ? 'opacity-0 w-0' : 'opacity-100 w-full'}
                      overflow-hidden 
                      whitespace-nowrap 
                      text-sm 
                      font-medium
                    `}
                  >
                    {item.label}
                  </span>
                </a>
              </div>
            ))}
        </nav>

        {/* Bottom Navigation */}
        <div className="border-t border-gray-700 p-2">
          <button
            onClick={handleLogout}
            className="
              flex items-center 
              px-4 py-3 
              hover:bg-red-600 
              text-gray-300 
              transition-all 
              duration-300 
              w-full 
              rounded-xl 
              group
            "
          >
            <MessageCircleQuestion className="w-5 h-5 group-hover:rotate-12 transition-transform"/>
            <span
              className={`
                ml-3 
                transition-all 
                ${isCollapsed ? 'opacity-0 w-0' : 'opacity-100 w-full'}
                overflow-hidden 
                whitespace-nowrap 
                text-m
                font-medium 
                group-hover:text-white
              `}
            >
              Logout
            </span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div
        className={`
          flex-auto 
          w-screen 
          h-screen 
          left-0 
          top-0 
          p-8 
          bg-gray-100 
          transition-all 
          duration-300 
          ${isCollapsed ? 'ml-16' : 'ml-64'}
        `}
      >
        {children}
      </div>
    </div>
  );
};

export default SideBarImageApp;
// components/ImageUpload.tsx
"use client";

import React, { useState, useRef } from 'react';
import { Upload, Trash2, Image as ImageIcon } from 'lucide-react';
import Image from 'next/image';

interface ImageUploadProps {
  onUpload: (files: File[], categories: string[]) => void;
}

const CATEGORIES = [
  'aGrass', 
  'bField', 
  'cIndustry', 
  'dRiverLake', 
  'eForest', 
  'fResident', 
  'gParking'
];

const ImageUpload: React.FC<ImageUploadProps> = ({ onUpload }) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<{[key: number]: string}>({});
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      const fileArray = Array.from(files);
      setSelectedFiles(prevFiles => [...prevFiles, ...fileArray]);
    }
  };

  const handleCategoryChange = (index: number, category: string) => {
    setSelectedCategories(prev => ({
      ...prev,
      [index]: category
    }));
  };

  const removeFile = (index: number) => {
    setSelectedFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
    const newCategories = {...selectedCategories};
    delete newCategories[index];
    setSelectedCategories(newCategories);
  };

  const handleUpload = () => {
    const categorizedFiles = selectedFiles.map((file, index) => {
      if (!selectedCategories[index]) {
        throw new Error(`Please select a category for ${file.name}`);
      }
      return file;
    });

    onUpload(categorizedFiles, categorizedFiles.map((_, index) => selectedCategories[index]));
    
    // Reset state after upload
    setSelectedFiles([]);
    setSelectedCategories({});
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white rounded-xl shadow-lg">
      <div 
        className="border-2 border-dashed border-blue-200 rounded-lg p-6 text-center 
        hover:border-blue-400 transition-colors duration-300"
        onClick={() => fileInputRef.current?.click()}
      >
        <input 
          type="file" 
          multiple 
          accept="image/*" 
          ref={fileInputRef}
          onChange={handleFileChange} 
          className="hidden"
        />
        <div className="flex flex-col items-center justify-center space-y-4">
          <Upload className="w-12 h-12 text-blue-500" />
          <p className="text-gray-600">
            Drag and drop images here or <span className="text-blue-500 cursor-pointer">browse</span>
          </p>
        </div>
      </div>

      {selectedFiles.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-4">Selected Images</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {selectedFiles.map((file, index) => (
              <div 
                key={index} 
                className="border rounded-lg p-4 flex flex-col space-y-4 relative"
              >
                <button 
                  onClick={() => removeFile(index)}
                  className="absolute top-2 right-2 text-red-500 hover:text-red-700"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
                
                <div className="flex justify-center items-center h-48 overflow-hidden rounded-lg">
                  <Image 
                    src={URL.createObjectURL(file)} 
                    alt={file.name}
                    width={200}
                    height={200}
                    className="object-cover"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Select Category
                  </label>
                  <select 
                    value={selectedCategories[index] || ''}
                    onChange={(e) => handleCategoryChange(index, e.target.value)}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200"
                  >
                    <option value="">Choose a category</option>
                    {CATEGORIES.map(category => (
                      <option key={category} value={category}>
                        {category}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 flex justify-end space-x-4">
            <button
              onClick={handleUpload}
              disabled={selectedFiles.length === 0 || 
                       Object.keys(selectedCategories).length !== selectedFiles.length}
              className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 
              transition-colors duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Upload Images
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
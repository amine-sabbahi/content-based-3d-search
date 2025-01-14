// component/ModelUpload.tsx
"use client";

import React, { useState, useRef } from "react";
import { Upload, Trash2 } from "lucide-react";
import { toast } from "react-hot-toast";

interface ModelUploadProps {
  onUpload: (file: File, category: string) => void;
  categories: string[];
}

const ModelUpload: React.FC<ModelUploadProps> = ({ onUpload, categories }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleCategoryChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedCategory(event.target.value);
  };

  const handleUpload = () => {
    if (!selectedFile) {
      toast.error("Please select a file to upload.");
      return;
    }

    if (!selectedCategory) {
      toast.error("Please select a category.");
      return;
    }

    onUpload(selectedFile, selectedCategory);
    setSelectedFile(null);
    setSelectedCategory("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white rounded-xl shadow-lg">
      <div
        className="border-2 border-dashed border-blue-200 rounded-lg p-6 text-center hover:border-blue-400 transition-colors duration-300"
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          accept=".obj"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
        />
        <div className="flex flex-col items-center justify-center space-y-4">
          <Upload className="w-12 h-12 text-blue-500" />
          <p className="text-gray-600">
            Drag and drop 3D models here or <span className="text-blue-500 cursor-pointer">browse</span>
          </p>
          <p className="text-sm text-gray-500">Allowed file types: .obj</p>
        </div>
      </div>

      {selectedFile && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-4">Selected File</h3>
          <div className="border rounded-lg p-4 flex flex-col space-y-4 relative">
            <button
              onClick={() => setSelectedFile(null)}
              className="absolute top-2 right-2 text-red-500 hover:text-red-700"
            >
              <Trash2 className="w-5 h-5" />
            </button>

            <div>
              <p className="text-sm font-medium text-gray-600 truncate">{selectedFile.name}</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Select Category</label>
              <select
                value={selectedCategory}
                onChange={handleCategoryChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200"
              >
                <option value="">Choose a category</option>
                {categories.map((category) => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </div>

            <div className="mt-6 flex justify-end space-x-4">
              <button
                onClick={handleUpload}
                disabled={!selectedFile || !selectedCategory}
                className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition-colors duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Upload Model
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelUpload;
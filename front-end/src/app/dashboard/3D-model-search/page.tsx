// dashboard/simple-return/page.tsx
"use client";

import { useState, useEffect } from 'react';
import SideBarAdmin from '@/components/SideBar';
import axios from 'axios';
import { toast, Toaster } from 'react-hot-toast';
import Image from 'next/image';
import { SearchResult, DESCRIPTOR_TYPES, DescriptorType } from '@/types/search';
import ImageSearchCard from '@/components/ImageSearchCard';

interface UploadedImage {
  filename: string;
  category: string;
  originalName: string;
}

const SimpleSimilaritySearch = () => {
  const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  // Search State
  const [selectedImage, setSelectedImage] = useState<UploadedImage | null>(null);
  const [descriptorType, setDescriptorType] = useState<DescriptorType>('all');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);

  // Fetch existing images
  const fetchExistingImages = async () => {
    try {
      const response = await axios.get('http://localhost:5000/get-existing-images');
      
      const existingImages: UploadedImage[] = response.data.images.map((image: any) => ({
        filename: image.filename,
        category: image.category,
        originalName: image.original_name
      }));

      setUploadedImages(existingImages);
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching existing images:', error);
      toast.error('Failed to load existing images');
      setIsLoading(false);
    }
  };

  // Perform image similarity search
  const performSearch = async () => {
    if (!selectedImage) {
      toast.error('Please select an image to search');
      return;
    }

    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append('image', selectedImage.filename);
      formData.append('category', selectedImage.category);
      formData.append('descriptorType', descriptorType);

      const response = await axios.post('http://localhost:5000/similarity-search', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setSearchResults(response.data.results);
      setIsLoading(false);
    } catch (error) {
      console.error('Search error:', error);
      toast.error('Failed to perform similarity search');
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchExistingImages();
  }, []);

  if (isLoading) {
    return (
      <SideBarAdmin>
        <div className="flex justify-center items-center h-screen">
          <p className="text-xl text-gray-600">Loading...</p>
        </div>
      </SideBarAdmin>
    );
  }

  return (
    <SideBarAdmin>
      <div className="flex h-screen">
        {/* Left Sidebar - Image Selection and Search Options */}
        <div className="w-1/4 p-4 border-r bg-gray-100 overflow-y-auto">
          <h2 className="text-xl font-bold mb-4">Image Similarity Search</h2>
          
          {/* Descriptor Type Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700">Descriptor Type</label>
            <select 
              value={descriptorType} 
              onChange={(e) => setDescriptorType(e.target.value as DescriptorType)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            >
              {DESCRIPTOR_TYPES.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>

          {/* Image Grid for Selection */}
          <div className="grid grid-cols-3 gap-2">
            {uploadedImages.map((image, index) => (
              <div 
                key={index} 
                className={`cursor-pointer border-2 ${
                  selectedImage?.filename === image.filename 
                    ? 'border-blue-500' 
                    : 'border-transparent'
                }`}
                onClick={() => setSelectedImage(image)}
              >
                <Image 
                  src={`http://localhost:5000/${image.category}/${image.filename}`} 
                  alt={image.originalName}
                  width={100}
                  height={100}
                  className="object-cover w-full h-full"
                />
              </div>
            ))}
          </div>

          {/* Search Button */}
          <button 
            onClick={performSearch}
            disabled={!selectedImage}
            className="mt-4 w-full bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300"
          >
            Search Similar Images
          </button>
        </div>

        {/* Right Side - Search Results */}
        <div className="w-3/4 p-4 overflow-y-auto">
          <h2 className="text-xl font-bold mb-4">Search Results</h2>
          <div className="grid grid-cols-4 gap-4">
            {searchResults.map((result, index) => (
              <ImageSearchCard 
                key={index} 
                result={result}
                descriptorType={descriptorType}
              />
            ))}
          </div>
        </div>
      </div>
      <Toaster />
    </SideBarAdmin>
  );
};

export default SimpleSimilaritySearch;
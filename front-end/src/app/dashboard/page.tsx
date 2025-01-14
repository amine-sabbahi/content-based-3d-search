// dashboard/page.tsx
"use client";

import { useState, useEffect } from 'react';
import SideBarAdmin from '@/components/SideBar';
import ImageUpload from '@/components/ImageUpload';
import axios from 'axios';
import { toast, Toaster } from 'react-hot-toast';
import Image from 'next/image';

interface UploadedImage {
  filename: string;
  category: string;
  originalName: string;
  localFile?: File;
}

const Dashboard = () => {
  const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // New function to fetch existing images
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

  // Load existing images when component mounts
  useEffect(() => {
    fetchExistingImages();
  }, []);

  const handleImageUpload = async (files: File[], categories: string[]) => {
    const formData = new FormData();
    
    files.forEach((file, index) => {
      formData.append('images', file);
      formData.append('categories', categories[index]);
    });

    try {
      const response = await axios.post('http://localhost:5000/upload-images', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      toast.success('Images uploaded successfully!');
      
      // Combine server response with local file data
      const newUploadedImages: UploadedImage[] = response.data.files.map((serverFile: any, index: number) => ({
        ...serverFile,
        localFile: files[index]
      }));

      setUploadedImages(prev => [...prev, ...newUploadedImages]);
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to upload images');
    }
  };

  const removeImage = (index: number) => {
    const imageToRemove = uploadedImages[index];
    
    // Optimistically remove the image from the UI
    const updatedImages = uploadedImages.filter((_, i) => i !== index);
    setUploadedImages(updatedImages);

    // Send delete request to backend
    axios.delete('http://localhost:5000/delete-image', {
      data: { 
        filename: imageToRemove.filename, 
        category: imageToRemove.category 
      }
    })
    .then(() => {
      toast.success('Image deleted successfully');
    })
    .catch((error) => {
      console.error('Delete error:', error);
      toast.error('Failed to delete image');
      // Revert the optimistic update if delete fails
      setUploadedImages(prev => [...prev, imageToRemove]);
    });
  };

  if (isLoading) {
    return (
      <SideBarAdmin>
        <div className="flex justify-center items-center h-screen">
          <p className="text-xl text-gray-600">Loading images...</p>
        </div>
      </SideBarAdmin>
    );
  }

  return (
    <SideBarAdmin>
  <Toaster position="top-right" />
  <div className="flex flex-col items-center justify-center h-screen space-y-6 max-w-6xl mx-auto">
    <h1 className="text-3xl font-bold text-gray-800 mb-4">Image Upload Dashboard</h1>

    <ImageUpload onUpload={handleImageUpload} />

    {uploadedImages.length > 0 && (
      <div className="w-full mt-8">
        <h2 className="text-2xl font-semibold text-gray-700 mb-4">
          Uploaded Images ({uploadedImages.length})
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
          {uploadedImages.map((image, index) => (
            <div 
              key={index} 
              className="bg-white rounded-lg shadow-md overflow-hidden relative group"
            >
              {/* Image Preview */}
              <div className="relative h-48 w-full">
                <Image 
                  src={image.localFile 
                    ? URL.createObjectURL(image.localFile) 
                    : `http://localhost:5000/${image.category}/${image.filename}`
                  } 
                  alt={image.originalName || 'Uploaded image'} 
                  fill
                  className="object-cover group-hover:opacity-70 transition-opacity"
                />
                
                {/* Delete Overlay */}
                <button 
                  onClick={() => removeImage(index)}
                  className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full 
                  opacity-0 group-hover:opacity-100 transition-opacity duration-300 
                  hover:bg-red-600 z-10"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
              
              {/* Image Details */}
              <div className="p-4">
                <p className="text-sm font-medium text-gray-600 truncate">
                  {image.originalName}
                </p>
                <p className="text-xs text-gray-500">
                  Category: {image.category}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
</SideBarAdmin>

  );
};

export default Dashboard;
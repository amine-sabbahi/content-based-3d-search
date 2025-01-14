// app/dashboard/feedback-return

"use client";

import { useState, useEffect } from 'react';
import SideBarAdmin from '@/components/SideBar';
import axios from 'axios';
import { toast, Toaster } from 'react-hot-toast';
import Image from 'next/image';
import { SearchResult, DESCRIPTOR_TYPES, DescriptorType } from '@/types/search';
import ImageSearchCard from '@/components/ImageSearchCard2';

interface UploadedImage {
  filename: string;
  category: string;
  originalName: string;
}

const FeedbackSimilaritySearch = () => {
  const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  // Search State
  const [selectedImage, setSelectedImage] = useState<UploadedImage | null>(null);
  
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [relevantImages, setRelevantImages] = useState<string[]>([]);
  const [irrelevantImages, setIrrelevantImages] = useState<string[]>([]);

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

  // Perform image initial search
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
      formData.append('originalName', selectedImage.originalName);
      

      const response = await axios.post('http://localhost:5000/initial-search', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setSearchResults(response.data.results);
      setRelevantImages([]);
      setIrrelevantImages([]);
      setIsLoading(false);
    } catch (error) {
      console.error('Search error:', error);
      toast.error('Failed to perform initial search');
      setIsLoading(false);
    }
  };

  // Update relevance feedback
  const updateRelevanceFeedback = async () => {
    try {
      const formData = new FormData();
      formData.append('query_image', selectedImage?.filename || '');
      formData.append('category', selectedImage?.category || '');
      formData.append('originalName', selectedImage?.originalName || '');
      formData.append('relevant_images', JSON.stringify(relevantImages));
      formData.append('irrelevant_images', JSON.stringify(irrelevantImages));

      const response = await axios.post('http://localhost:5000/relevance-feedback', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      // Update search results with refined search
      setSearchResults(response.data.results);
      toast.success('Relevance feedback applied');
      setRelevantImages([]);
      setIrrelevantImages([]);
    } catch (error) {
      console.error('Feedback error:', error);
      toast.error('Failed to apply relevance feedback');
    }
  };

  // Toggle image selection for relevance
  const toggleRelevance = (imagePath: string, type: 'relevant' | 'irrelevant') => {
    if (type === 'relevant') {
      setRelevantImages(prev => 
        prev.includes(imagePath) 
          ? prev.filter(img => img !== imagePath)
          : [...prev, imagePath]
      );
      setIrrelevantImages(prev => prev.filter(img => img !== imagePath));
    } else {
      setIrrelevantImages(prev => 
        prev.includes(imagePath)
          ? prev.filter(img => img !== imagePath)
          : [...prev, imagePath]
      );
      setRelevantImages(prev => prev.filter(img => img !== imagePath));
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
          


          {/* Image Grid for Selection */}
          <div className="grid grid-cols-3 gap-2">
            {uploadedImages.map((image) => (
              <div 
                key={image.filename} 
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
          {searchResults.length > 0 ? (
            <div className="grid grid-cols-4 gap-4">
              {searchResults.map((result) => (
                <ImageSearchCard
                  key={result.imagePath}
                  imagePath={result.imagePath}
                  originalName={result.originalName}
                  distance={result.distance}
                  category={result.category}
                  onRelevant={() => toggleRelevance(result.imagePath, 'relevant')}
                  onIrrelevant={() => toggleRelevance(result.imagePath, 'irrelevant')}
                  isRelevant={relevantImages.includes(result.imagePath)}
                  isIrrelevant={irrelevantImages.includes(result.imagePath)}
                />
              ))}
            </div>
          ) : (
            <div className="flex justify-center items-center h-full text-gray-500">
              No search results. Select an image and perform a search.
            </div>
          )}

          {/* Relevance Feedback Button */}
          {searchResults.length > 0 && (
            <div className="mt-4 flex justify-center">
              <button 
                onClick={updateRelevanceFeedback}
                disabled={relevantImages.length === 0 && irrelevantImages.length === 0}
                className="px-6 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                Apply Relevance Feedback
              </button>
            </div>
          )}
        </div>
      </div>
      <Toaster />
    </SideBarAdmin>
  );
};

export default FeedbackSimilaritySearch;
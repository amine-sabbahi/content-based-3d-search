// dashboard/simple-return/page.tsx
"use client";

import { useState, useEffect } from 'react';
import SideBarAdmin from '@/components/SideBar';
import axios from 'axios';
import { toast, Toaster } from 'react-hot-toast';
import Image from 'next/image';
import { SearchResult, DESCRIPTOR_TYPES, DescriptorType } from '@/types/search';
import ImageSearchCard from '@/components/ImageSearchCard';

interface UploadedModel {
  filename: string;
  thumbnailUrl: string | null;
  category: string;
}


const SimpleSimilaritySearch = () => {
  const [uploadedModels, setUploadedModels] = useState<UploadedModel[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  // Search State
  const [selectedImage, setSelectedImage] = useState<UploadedModel | null>(null);
  const [descriptorType, setDescriptorType] = useState<DescriptorType>('regular');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);

  const getDatasetPath = (type: DescriptorType): string => {
    switch (type) {
      case 'reduction_20':
        return 'dataset_features_reduced_20.csv';
      case 'reduction_50':
        return 'dataset_features_reduced_50.csv';
      case 'reduction_70':
        return 'dataset_features_reduced_70.csv';
      default:
        return 'dataset_features.csv';
    }
  };

  // Fetch uploaded models
  const fetchUploadedModels = async () => {
    try {
      const response = await axios.get("http://localhost:5000/get-uploaded-models");
      const models = response.data.models.map((model: UploadedModel) => ({
        ...model,
        thumbnailUrl: model.thumbnailUrl ? `http://localhost:5000${model.thumbnailUrl}` : null,
      }));
      setUploadedModels(models);
      setIsLoading(false);
    } catch (error) {
      console.error("Error fetching uploaded models:", error);
      toast.error("Failed to load uploaded models");
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
      formData.append('datasetPath', getDatasetPath(descriptorType));

      if (descriptorType !== 'regular') {
        const reductionFactor = parseFloat(descriptorType.split('_')[1]) / 100;
        formData.append('reductionFactor', reductionFactor.toString());
      }

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
    fetchUploadedModels();
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
          <h2 className="text-xl font-bold mb-4">Model Similarity Search</h2>
          
          {/* Descriptor Type Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700">Type</label>
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
            {uploadedModels.map((model, index) => (
              <div 
                key={index} 
                className={`cursor-pointer border-2 ${
                  selectedImage?.filename === model.filename 
                    ? 'border-blue-500' 
                    : 'border-transparent'
                }`}
                onClick={() => setSelectedImage(model)}
              > {model.thumbnailUrl ? (
                 <Image 
                   src={model.thumbnailUrl}
                   alt={model.filename}
                   width={100}
                   height={100}
                   className="object-cover w-full h-full"
                 />) : (
                  <div className="flex items-center justify-center h-full bg-gray-100">
                    <p className="text-gray-500">No Thumbnail</p>
                    
                  </div>
                  
                )}
              </div>
            ))}
          </div>

          {/* Search Button */}
          <button 
            onClick={performSearch}
            disabled={!selectedImage}
            className="mt-4 w-full bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300"
          >
            Search Similar Models
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
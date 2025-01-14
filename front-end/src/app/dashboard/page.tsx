// dashboard/page.tsx

"use client";

import { useState, useEffect } from "react";
import SideBarAdmin from "@/components/SideBar";
import ModelUpload from "@/components/ModelUpload";
import axios from "axios";
import { toast, Toaster } from "react-hot-toast";
import Image from "next/image";
import { Trash2 } from "lucide-react";

interface UploadedModel {
  filename: string;
  thumbnailUrl: string | null;
  category: string;
}

const CATEGORIES = ['Abstract', 'Alabastron', 'All Models', 'Amphora', 'Aryballos', 'Bowl', 'Dinos', 'Hydria', 'Kalathos', 'Kantharos', 'Krater', 'Kyathos', 'Kylix', 'Lagynos', 'Lebes', 'Lekythos', 'Lydion', 'Mastos', 'Modern-Bottle', 'Modern-Glass', 'Modern-Mug', 'Modern-Vase', 'Mug', 'Native American - Bottle', 'Native American - Bowl', 'Native American - Effigy', 'Native American - Jar', 'Nestoris', 'Oinochoe', 'Other', 'Pelike', 'Picher Shaped', 'Pithoeidi', 'Pithos', 'Psykter', 'Pyxis', 'Skyphos']

const Dashboard = () => {
  const [uploadedModels, setUploadedModels] = useState<UploadedModel[]>([]);
  const [isLoading, setIsLoading] = useState(true);

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

  useEffect(() => {
    fetchUploadedModels();
  }, []);

  // Handle file upload
  const handleFileUpload = async (file: File, category: string) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("category", category);

    try {
      const response = await axios.post("http://localhost:5000/upload-model", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      toast.success(response.data.message);
      fetchUploadedModels(); // Refresh the list of uploaded models
    } catch (error) {
      console.error("Upload error:", error);
      toast.error("Failed to upload model");
    }
  };

  // Handle model deletion
  const handleDeleteModel = async (filename: string) => {
    try {
      await axios.delete("http://localhost:5000/delete-model", {
        data: { filename },
      });

      toast.success("Model deleted successfully");
      fetchUploadedModels(); // Refresh the list of uploaded models
    } catch (error) {
      console.error("Delete error:", error);
      toast.error("Failed to delete model");
    }
  };

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
      <Toaster position="top-right" />
      <div className="flex flex-col items-center justify-center space-y-6 max-w-6xl mx-auto p-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-4">Dashboard</h1>

        {/* File Upload Section */}
        <div className="w-full">
          <h2 className="text-2xl font-semibold text-gray-700 mb-4">Upload 3D Model</h2>
          <ModelUpload onUpload={handleFileUpload} categories={CATEGORIES} />
        </div>

        {/* Display Uploaded Models */}
        {uploadedModels.length > 0 && (
          <div className="w-full mt-8">
            <h2 className="text-2xl font-semibold text-gray-700 mb-4">
              Uploaded Models ({uploadedModels.length})
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-6">
              {uploadedModels.map((model, index) => (
                <div key={index} className="bg-white rounded-lg shadow-md overflow-hidden relative group">
                  <div className="relative h-48 w-full">
                    {model.thumbnailUrl ? (
                      <Image
                        src={model.thumbnailUrl}
                        alt={`Thumbnail for ${model.filename}`}
                        fill
                        className="object-cover group-hover:opacity-70 transition-opacity"
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full bg-gray-100">
                        <p className="text-gray-500">No Thumbnail</p>
                      </div>
                    )}
                    <button
                      onClick={() => handleDeleteModel(model.filename)}
                      className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 hover:bg-red-600 z-10"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="p-4">
                    <p className="text-sm font-medium text-gray-600 truncate">{model.filename}</p>
                    <p className="text-xs text-gray-500">Category: {model.category}</p>
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
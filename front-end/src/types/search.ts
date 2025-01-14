// types/search.ts
export interface SearchResult {
    imagePath: string;
    distance: number;
    category: string;
    originalName: string;
  }
  
  export const DESCRIPTOR_TYPES = [
    'all', 
    'histogram_color', 
    'co-occurrence', 
    'hog', 
    'lbp', 
    'hu', 
    'gabor', 
    'dominant_colors'
  ] as const;
  
  export type DescriptorType = typeof DESCRIPTOR_TYPES[number];
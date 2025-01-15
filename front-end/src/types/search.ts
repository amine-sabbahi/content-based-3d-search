// types/search.ts
export interface SearchResult {
    imagePath: string;
    distance: number;
    category: string;
    originalName: string;
    Thumbnail: string;
  }
  
  export const DESCRIPTOR_TYPES = [
    'regular',
    'reduction_20',
    'reduction_50',
    'reduction_70'
  ] as const;
  
  export type DescriptorType = typeof DESCRIPTOR_TYPES[number];
export interface BuildingInfo {
  lat: number
  lon: number
  address: string | null
  age: {
    classId: number
    interval: string
    confidence: number // 0 to 1
  } | null // null = no age model for this area
  size: {
    areaM2: number | null
    heightM: number | null
  }
  land: {
    elevationM: number | null
    surroundings: string | null
  }
  surface: string | null
}
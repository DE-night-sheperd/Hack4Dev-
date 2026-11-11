import type { BuildingInfo } from './types'

// Rough boxes around the two cities in the dataset (replace with the real data extent later)
const CITIES = [
  { name: 'Madrid', south: 40.3, north: 40.55, west: -3.9, east: -3.5 },
  { name: 'Amsterdam', south: 52.2, north: 52.52, west: 4.65, east: 5.15 },
]

function inModelArea(lat: number, lon: number) {
  return CITIES.some(
    (c) => lat >= c.south && lat <= c.north && lon >= c.west && lon <= c.east
  )
}

async function reverseGeocode(lat: number, lon: number): Promise<string | null> {
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/reverse?format=json&zoom=18&lat=${lat}&lon=${lon}`
    )
    if (!res.ok) return null
    const d = await res.json()
    return d.display_name ?? null
  } catch {
    return null
  }
}

export async function getBuildingInfo(lat: number, lon: number): Promise<BuildingInfo> {
  const address = await reverseGeocode(lat, lon)

  return {
    lat,
    lon,
    address,
    // Placeholder prediction, shown only inside the two model areas
    age: inModelArea(lat, lon)
      ? { classId: 3, interval: '1984–2004', confidence: 0.62 }
      : null,
    size: { areaM2: null, heightM: null },
    land: { elevationM: null, surroundings: null },
    surface: null,
  }
}
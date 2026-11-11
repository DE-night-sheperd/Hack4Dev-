import { useState } from 'react'
import * as Cesium from 'cesium'

interface Hit {
  name: string
  lat: number
  lon: number
}

const COORDS = /^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/

export default function SearchBar({ viewer }: { viewer: Cesium.Viewer | null }) {
  const [query, setQuery] = useState('')
  const [message, setMessage] = useState('')
  const [hits, setHits] = useState<Hit[]>([])

  async function search() {
    if (!viewer || !query.trim()) return
    setHits([])

    // Coordinates typed directly: always works
    const m = query.match(COORDS)
    if (m) {
      const lat = Number(m[1])
      const lon = Number(m[2])
      if (Math.abs(lat) <= 90 && Math.abs(lon) <= 180) {
        await goTo({ name: 'Typed coordinates', lat, lon })
        return
      }
    }

    setMessage('Searching...')
    try {
      const q = query.trim().replace(/,\s*$/, '')
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&limit=5&q=${encodeURIComponent(q)}`
      )
      const data: { display_name: string; lat: string; lon: string }[] = await res.json()
      if (data.length === 0) {
        setMessage('Place not found. Try adding the city, or type coordinates like -26.18, 28.04.')
        return
      }
      const found = data.map((d) => ({
        name: d.display_name,
        lat: Number(d.lat),
        lon: Number(d.lon),
      }))
      if (found.length === 1) {
        await goTo(found[0])
      } else {
        setMessage('Pick the right match:')
        setHits(found)
      }
    } catch {
      setMessage('Search failed. Try again.')
    }
  }

  async function goTo(h: Hit) {
    if (!viewer) return

    viewer.entities.removeById('search-pin')
    viewer.entities.add({
      id: 'search-pin',
      position: Cesium.Cartesian3.fromDegrees(h.lon, h.lat),
      point: {
        pixelSize: 14,
        color: Cesium.Color.YELLOW,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    })

    let ground = 0
    try {
      const [pos] = await Cesium.sampleTerrainMostDetailed(viewer.terrainProvider, [
        Cesium.Cartographic.fromDegrees(h.lon, h.lat),
      ])
      ground = pos.height ?? 0
    } catch {
      ground = 0
    }

    viewer.camera.flyToBoundingSphere(
      new Cesium.BoundingSphere(Cesium.Cartesian3.fromDegrees(h.lon, h.lat, ground), 1),
      {
        offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-35), 1200),
        duration: 3,
      }
    )
    setHits([])
    setMessage(
      `${h.name} (${h.lat.toFixed(5)}, ${h.lon.toFixed(5)}). Address text is approximate (OpenStreetMap).`
    )
  }

  return (
    <div className="search-bar">
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && search()}
        placeholder="Search a place, address, or type lat, lon"
      />
      <button onClick={search}>Search</button>
      {message && <div className="search-message">{message}</div>}
      {hits.length > 0 && (
        <ul className="search-results">
          {hits.map((h, i) => (
            <li key={i}>
              <button onClick={() => goTo(h)}>
                {h.name}
                <small> ({h.lat.toFixed(5)}, {h.lon.toFixed(5)})</small>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
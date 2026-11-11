import { useEffect, useRef, useState } from 'react'
import * as Cesium from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'
import SearchBar from './components/SearchBar'
import MapControls from './components/MapControls'
import InfoCard from './components/InfoCard'
import SpectralPanel from './components/SpectralPanel'
import { addPlaceLabels } from './labels'
import { getBuildingInfo } from './api'
import type { BuildingInfo } from './types'

Cesium.Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_TOKEN

export default function App() {
  const containerRef = useRef<HTMLDivElement>(null)
  const tilesetRef = useRef<Cesium.Cesium3DTileset | null>(null)
  const [viewer, setViewer] = useState<Cesium.Viewer | null>(null)
  const [info, setInfo] = useState<BuildingInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(false)
  const [cardOpen, setCardOpen] = useState(false)

  useEffect(() => {
    if (!containerRef.current) return

    const v = new Cesium.Viewer(containerRef.current, {
      timeline: false,
      animation: false,
      baseLayerPicker: false,
      geocoder: false,
      terrain: Cesium.Terrain.fromWorldTerrain(),
    })
    v.resolutionScale = window.devicePixelRatio

    Cesium.createGooglePhotorealistic3DTileset()
      .then((tileset) => {
        tileset.maximumScreenSpaceError = 6
        tilesetRef.current = tileset
        v.scene.primitives.add(tileset)
      })
      .catch((err) => console.error('3D tiles failed to load', err))

    const removeLabels = addPlaceLabels(v)

    const handler = new Cesium.ScreenSpaceEventHandler(v.scene.canvas)
    handler.setInputAction(async (click: { position: Cesium.Cartesian2 }) => {
      const point = v.scene.pickPosition(click.position)
      if (!Cesium.defined(point)) return

      const c = Cesium.Cartographic.fromCartesian(point)
      const lat = Cesium.Math.toDegrees(c.latitude)
      const lon = Cesium.Math.toDegrees(c.longitude)

      v.entities.removeById('click-pin')
      v.entities.add({
        id: 'click-pin',
        position: point,
        point: {
          pixelSize: 12,
          color: Cesium.Color.RED,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      })

      setCardOpen(true)
      setLoading(true)
      setError(false)
      try {
        setInfo(await getBuildingInfo(lat, lon))
      } catch {
        setError(true)
      }
      setLoading(false)
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK)

    setViewer(v)
    return () => {
      handler.destroy()
      removeLabels()
      setViewer(null)
      v.destroy()
    }
  }, [])

  return (
    <>
      <div ref={containerRef} className="map" />
      <SearchBar viewer={viewer} />
      <MapControls viewer={viewer} />
      <SpectralPanel viewer={viewer} tilesetRef={tilesetRef} />
      {cardOpen && (
        <InfoCard
          info={info}
          loading={loading}
          error={error}
          onClose={() => {
            setCardOpen(false)
            viewer?.entities.removeById('click-pin')
          }}
        />
      )}
    </>
  )
}
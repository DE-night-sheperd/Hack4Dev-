import { useRef, useState } from 'react'
import * as Cesium from 'cesium'

const LAYERS = [
  { id: 'off', label: 'No spectral layer' },
  { id: 'lst', label: 'Surface temperature (°C)' },
  { id: 'heat', label: 'Heat anomaly (vs surroundings)' },
  { id: 'truecolor', label: 'True colour' },
  { id: 'falsecolor', label: 'False colour (vegetation = red)' },
  { id: 'ndvi', label: 'Greenness (NDVI)' },
  { id: 'ndbi', label: 'Built-up index (NDBI)' },
]

const THERMAL = ['lst', 'heat']
const INDEX = ['lst', 'heat', 'ndvi', 'ndbi']

export default function SpectralPanel({
  viewer,
  tilesetRef,
  labelLayerRef,
}: {
  viewer: Cesium.Viewer | null
  tilesetRef: React.RefObject<Cesium.Cesium3DTileset | null>
  labelLayerRef?: React.RefObject<Cesium.ImageryLayer | null>
}) {
  const layerRef = useRef<Cesium.ImageryLayer | null>(null)
  const homeRef = useRef<Cesium.ImageryLayerCollection | null>(null)
  const [value, setValue] = useState('off')
  const [opacity, setOpacity] = useState(0.7)
  const [labelsOn, setLabelsOn] = useState(true)
  const [drape, setDrape] = useState(false)
  const [msg, setMsg] = useState('')

  function clearLayer() {
    if (layerRef.current && homeRef.current) {
      homeRef.current.remove(layerRef.current)
    }
    layerRef.current = null
    homeRef.current = null
  }

  async function choose(id: string, useDrape: boolean = drape) {
    if (!viewer) return
    setValue(id)
    clearLayer()

    if (id === 'off') {
      if (tilesetRef.current) tilesetRef.current.show = true
      setMsg('')
      return
    }

    const canvas = viewer.scene.canvas
    const centre = new Cesium.Cartesian2(canvas.clientWidth / 2, canvas.clientHeight / 2)
    const hit = viewer.camera.pickEllipsoid(centre) ?? viewer.camera.positionWC
    const c = Cesium.Cartographic.fromCartesian(hit)
    const lat = Cesium.Math.toDegrees(c.latitude)
    const lon = Cesium.Math.toDegrees(c.longitude)

    setMsg('Loading layer (can take 10 to 20 seconds)...')
    try {
      const res = await fetch(
        `http://localhost:8000/ee/tiles?layer=${id}&lat=${lat}&lon=${lon}`
      )
      if (!res.ok) throw new Error('bad response ' + res.status)
      const { url } = await res.json()
      console.log('Tile URL template:', url)

      const provider = new Cesium.UrlTemplateImageryProvider({
        url,
        maximumLevel: THERMAL.includes(id) ? 11 : 14,
        credit: 'Satellite data via Google Earth Engine',
      })
      provider.errorEvent.addEventListener((e: any) => {
        console.warn('Layer tile error:', e)
        setMsg('Some layer tiles failed to load. Open the Console (F12) for details.')
      })

      const options: any = { alpha: opacity }
      if (INDEX.includes(id)) {
        options.magnificationFilter = Cesium.TextureMagnificationFilter.NEAREST
      }
      const layer = new Cesium.ImageryLayer(provider, options)

      const tileset = tilesetRef.current
      if (useDrape && tileset?.imageryLayers) {
        tileset.imageryLayers.add(layer, 0)
        homeRef.current = tileset.imageryLayers
        tileset.show = true
      } else {
        viewer.imageryLayers.add(layer)
        homeRef.current = viewer.imageryLayers
        if (tileset) tileset.show = false // hide 3D so the globe layer is visible
      }
      layerRef.current = layer

      setMsg(
        THERMAL.includes(id)
          ? 'Summer surface heat from Landsat (100 m pixels). Area context only. It does not measure building insulation.'
          : 'Sentinel-2 (10 m pixels). Covers about 30 km around the centre of the screen.'
      )
    } catch (e) {
      console.error('Layer failed', e)
      setMsg('Could not load the layer. Check the Python service and the Console (F12).')
    }
  }

  function changeOpacity(v: number) {
    setOpacity(v)
    if (layerRef.current) layerRef.current.alpha = v
  }

  function toggleLabels(on: boolean) {
    setLabelsOn(on)
    if (labelLayerRef?.current) labelLayerRef.current.show = on
  }

  function toggleDrape(on: boolean) {
    setDrape(on)
    if (value !== 'off') choose(value, on)
  }

  return (
    <div className="spectral-panel">
      <b>Layers</b>
      <select value={value} onChange={(e) => choose(e.target.value)}>
        {LAYERS.map((l) => (
          <option key={l.id} value={l.id}>{l.label}</option>
        ))}
      </select>
      {value !== 'off' && (
        <>
          <label>
            Opacity {Math.round(opacity * 100)}%
            <input
              type="range"
              min="0.2"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(e) => changeOpacity(Number(e.target.value))}
            />
          </label>
          <label className="row">
            <input
              type="checkbox"
              checked={drape}
              onChange={(e) => toggleDrape(e.target.checked)}
            />
            Draw over 3D buildings (experimental)
          </label>
        </>
      )}
      {labelLayerRef && (
        <label className="row">
          <input
            type="checkbox"
            checked={labelsOn}
            onChange={(e) => toggleLabels(e.target.checked)}
          />
          Street and place names
        </label>
      )}
      {msg && <small>{msg}</small>}
    </div>
  )
}
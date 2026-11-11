import * as Cesium from 'cesium'

export default function MapControls({ viewer }: { viewer: Cesium.Viewer | null }) {
  if (!viewer) return null

  const cam = viewer.camera
  const zoom = (factor: number) => {
    const h = cam.positionCartographic.height
    cam.zoomIn(h * factor)
  }
  const tilt = (deg: number) => {
    cam.lookUp(Cesium.Math.toRadians(deg))
  }
  const rotate = (deg: number) => {
    cam.lookRight(Cesium.Math.toRadians(deg))
  }

  return (
    <div className="map-controls">
      <button onClick={() => zoom(0.3)}>＋ Zoom in</button>
      <button onClick={() => zoom(-0.3)}>－ Zoom out</button>
      <button onClick={() => tilt(10)}>▲ Tilt up</button>
      <button onClick={() => tilt(-10)}>▼ Tilt down</button>
      <button onClick={() => rotate(-15)}>⟲ Turn left</button>
      <button onClick={() => rotate(15)}>⟳ Turn right</button>
    </div>
  )
}
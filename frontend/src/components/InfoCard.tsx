import type { BuildingInfo } from '../types'

const show = (v: string | number | null, unit = '') =>
  v === null ? 'Unknown' : `${v}${unit}`

export default function InfoCard({
  info,
  loading,
  error,
  onClose,
}: {
  info: BuildingInfo | null
  loading: boolean
  error: boolean
  onClose: () => void
}) {
  return (
    <div className="info-card">
      <button className="close" onClick={onClose}>×</button>
      {loading && <p>Loading...</p>}
      {error && <p className="warn">Could not load data. Try again.</p>}
      {!loading && !error && info && (
        <>
          <h3>Location</h3>
          <p>
            {info.lat.toFixed(5)}, {info.lon.toFixed(5)}
          </p>
          <p>
            <a
              href={`https://www.google.com/maps?q=${info.lat},${info.lon}`}
              target="_blank"
              rel="noreferrer"
            >
              Open in Google Maps
            </a>
          </p>
          <p>{info.address ?? 'No address found'}</p>
          <small>
            Nearest address from OpenStreetMap. It is approximate and can be wrong or
            missing. Trust the coordinates.
          </small>

          <h3>Age</h3>
          {info.age ? (
            <>
              <p><b>Class {info.age.classId}</b> ({info.age.interval})</p>
              <p>{Math.round(info.age.confidence * 100)}% sure</p>
              {info.age.confidence < 0.5 && <p className="warn">Low confidence</p>}
              <small>
                Sample data until the model is connected. Predicted for the 30 m area
                around this point, not the building alone.
              </small>
            </>
          ) : (
            <p>Age not available here (model not adapted to this city)</p>
          )}

          <h3>Size</h3>
          <p>Footprint area: {show(info.size.areaM2, ' m²')}</p>
          <p>Height: {show(info.size.heightM, ' m')}</p>

          <h3>Land</h3>
          <p>Elevation: {show(info.land.elevationM, ' m')}</p>
          <p>Surroundings: {show(info.land.surroundings)}</p>

          <h3>Surface characteristics</h3>
          <p>{show(info.surface)}</p>
          <small>Based on satellite data. This does not identify the building material.</small>
        </>
      )}
    </div>
  )
}
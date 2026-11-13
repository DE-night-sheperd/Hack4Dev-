import * as Cesium from 'cesium'

interface OsmNode {
  id: number
  lat: number
  lon: number
  tags?: Record<string, string>
}

export function addPlaceLabels(viewer: Cesium.Viewer) {
  const source = new Cesium.CustomDataSource('place-labels')
  viewer.dataSources.add(source)
  viewer.creditDisplay.addStaticCredit(new Cesium.Credit('Labels © OpenStreetMap contributors'))

  const seen = new Set<number>()
  let timer: number | undefined
  let controller: AbortController | undefined

  async function load() {
    // Only load labels when zoomed in on an area (below about 20 km)
    if (viewer.camera.positionCartographic.height > 20000) {
      source.entities.removeAll()
      seen.clear()
      return
    }
    const rect = viewer.camera.computeViewRectangle()
    if (!rect) return

    let south = Cesium.Math.toDegrees(rect.south)
    let north = Cesium.Math.toDegrees(rect.north)
    let west = Cesium.Math.toDegrees(rect.west)
    let east = Cesium.Math.toDegrees(rect.east)

    // Keep the search area small so the request stays fast
    const half = 0.04
    const cLat = (south + north) / 2
    const cLon = (west + east) / 2
    south = Math.max(south, cLat - half)
    north = Math.min(north, cLat + half)
    west = Math.max(west, cLon - half)
    east = Math.min(east, cLon + half)

    const box = `${south},${west},${north},${east}`
    const query = `[out:json][timeout:15];(
      node["place"~"city|town|suburb|neighbourhood|quarter"]["name"](${box});
      node["tourism"~"attraction|museum|viewpoint"]["name"](${box});
      node["historic"~"monument|castle|memorial"]["name"](${box});
    );out 100;`

    controller?.abort()
    controller = new AbortController()

    try {
      const res = await fetch('https://overpass-api.de/api/interpreter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'data=' + encodeURIComponent(query),
        signal: controller.signal,
      })
      if (!res.ok) return
      const data = await res.json()

      if (seen.size > 500) {
        source.entities.removeAll()
        seen.clear()
      }

      for (const n of data.elements as OsmNode[]) {
        const name = n.tags?.['name:en'] ?? n.tags?.name
        if (!name || seen.has(n.id)) continue
        seen.add(n.id)

        const isPlace = Boolean(n.tags?.place)
        source.entities.add({
          position: Cesium.Cartesian3.fromDegrees(n.lon, n.lat),
          label: {
            text: name,
            font: isPlace ? 'bold 16px sans-serif' : '13px sans-serif',
            fillColor: isPlace ? Cesium.Color.WHITE : Cesium.Color.YELLOW,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 3,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 25000),
          },
        })
      }
    } catch {
      // Request cancelled or failed: keep the labels we already have
    }
  }

  const onMoveEnd = () => {
    window.clearTimeout(timer)
    timer = window.setTimeout(load, 600)
  }
  viewer.camera.moveEnd.addEventListener(onMoveEnd)

  // Call this when the map is removed
  return () => {
    window.clearTimeout(timer)
    controller?.abort()
    viewer.camera.moveEnd.removeEventListener(onMoveEnd)
    viewer.dataSources.remove(source)
  }
}
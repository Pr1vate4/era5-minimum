export type CoastlinePoint = {
  lat: number
  lng: number
}

export type CoastlinePath = {
  points: CoastlinePoint[]
}

type GeoJsonLineString = {
  type: 'LineString'
  coordinates: number[][]
}

type GeoJsonMultiLineString = {
  type: 'MultiLineString'
  coordinates: number[][][]
}

type CoastlineGeoJson = {
  type: 'FeatureCollection'
  features: Array<{
    type: 'Feature'
    geometry: GeoJsonLineString | GeoJsonMultiLineString | null
    properties?: Record<string, unknown>
  }>
}

export function convertGeoJsonToCoastlinePaths(
  value: unknown,
): CoastlinePath[] {
  const geoJson = parseCoastlineGeoJson(value)
  const paths: CoastlinePath[] = []

  for (const feature of geoJson.features) {
    const geometry = feature.geometry

    if (!geometry) {
      continue
    }

    const lines =
      geometry.type === 'LineString'
        ? [geometry.coordinates]
        : geometry.coordinates

    for (const line of lines) {
      const points: CoastlinePoint[] = []

      for (const coordinate of line) {
        const longitude = coordinate[0]
        const latitude = coordinate[1]

        if (
          !Number.isFinite(longitude) ||
          !Number.isFinite(latitude)
        ) {
          continue
        }

        points.push({
          lat: latitude,
          lng: longitude,
        })
      }

      if (points.length >= 2) {
        paths.push({
          points,
        })
      }
    }
  }

  return splitPathsAtAntimeridian(paths)
}

function parseCoastlineGeoJson(value: unknown): CoastlineGeoJson {
  if (!isRecord(value) || value.type !== 'FeatureCollection' || !Array.isArray(value.features)) {
    throw new Error('Coastline GeoJSON must be a FeatureCollection.')
  }

  const features: CoastlineGeoJson['features'] = []
  for (const candidate of value.features) {
    if (!isRecord(candidate) || candidate.type !== 'Feature') continue
    if (candidate.geometry === null) {
      features.push({ type: 'Feature', geometry: null })
      continue
    }
    if (!isRecord(candidate.geometry)) continue

    if (
      candidate.geometry.type === 'LineString' &&
      isCoordinateLine(candidate.geometry.coordinates)
    ) {
      features.push({
        type: 'Feature',
        geometry: {
          type: 'LineString',
          coordinates: candidate.geometry.coordinates,
        },
      })
      continue
    }

    if (
      candidate.geometry.type === 'MultiLineString' &&
      Array.isArray(candidate.geometry.coordinates) &&
      candidate.geometry.coordinates.every(isCoordinateLine)
    ) {
      features.push({
        type: 'Feature',
        geometry: {
          type: 'MultiLineString',
          coordinates: candidate.geometry.coordinates,
        },
      })
    }
  }

  return { type: 'FeatureCollection', features }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isCoordinateLine(value: unknown): value is number[][] {
  return (
    Array.isArray(value) &&
    value.every(
      (coordinate) =>
        Array.isArray(coordinate) &&
        coordinate.length >= 2 &&
        typeof coordinate[0] === 'number' &&
        typeof coordinate[1] === 'number',
    )
  )
}

function splitPathsAtAntimeridian(
  paths: CoastlinePath[],
): CoastlinePath[] {
  const result: CoastlinePath[] = []

  for (const path of paths) {
    let currentPoints: CoastlinePoint[] = []

    for (const point of path.points) {
      const previousPoint =
        currentPoints[currentPoints.length - 1]

      const crossesAntimeridian =
        previousPoint &&
        Math.abs(point.lng - previousPoint.lng) > 180

      if (crossesAntimeridian) {
        if (currentPoints.length >= 2) {
          result.push({
            points: currentPoints,
          })
        }

        currentPoints = []
      }

      currentPoints.push(point)
    }

    if (currentPoints.length >= 2) {
      result.push({
        points: currentPoints,
      })
    }
  }

  return result
}

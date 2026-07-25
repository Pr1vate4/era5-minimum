import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js'
import {
  convertGeoJsonToCoastlinePaths,
  type CoastlinePath,
} from '../data/coastlineUtils'
import { useElementSize } from '../hooks/useElementSize'
import type { GlobeCoordinates } from '../types/globe'
import { resolvePublicAssetUrl } from '../utils/resolvePublicAssetUrl'


type GlobeCanvasProps = {
  textureUrl?: string
  textureRevision: number
  autoRotate: boolean
  isFullscreen: boolean
  resetSignal: number
  onPointSelect: (coordinates: GlobeCoordinates) => void
  onTextureLoadingChange: (loading: boolean) => void
  onTextureError: (message: string | null) => void
}

type TextureCacheEntry = {
  promise: Promise<THREE.Texture>
  texture?: THREE.Texture
}

type Runtime = {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  materials: [THREE.MeshPhongMaterial, THREE.MeshPhongMaterial]
  spheres: [THREE.Mesh, THREE.Mesh]
  marker: THREE.Group
  activeSphere: 0 | 1
  currentTextureUrl?: string
  autoRotateRequested: boolean
  resumeTimer?: number
  animationFrame?: number
  visible: boolean
  disposed: boolean
}

const textureCache = new Map<string, TextureCacheEntry>()
const activeTextureUrls = new Set<string>()
const MAX_TEXTURE_CACHE_SIZE = 8
const LAND_DISPLACEMENT_SCALE = 0.050 //подьем суши
const LAND_BUMP_SCALE = 0.018
const COASTLINE_RADIUS = 1 + LAND_DISPLACEMENT_SCALE + 0.003
const MARKER_RADIUS = 1 + LAND_DISPLACEMENT_SCALE + 0.025
const COASTLINES_URL = 'data/globe/coastlines.geojson'
let coastlinePathsPromise: Promise<CoastlinePath[]> | undefined

function loadTexture(url: string) {
  const cached = textureCache.get(url)
  if (cached) return cached.promise

  const loader = new THREE.TextureLoader()
  const entry: TextureCacheEntry = {
    promise: loader.loadAsync(url).then((texture) => {
      texture.colorSpace = THREE.SRGBColorSpace
      configureEquirectangularTexture(texture)
      texture.anisotropy = 8
      entry.texture = texture
      pruneTextureCache(url)
      return texture
    }),
  }

  entry.promise.catch(() => textureCache.delete(url))
  textureCache.set(url, entry)
  return entry.promise
}

function configureEquirectangularTexture(texture: THREE.Texture) {
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.ClampToEdgeWrapping
  // SphereGeometry starts its U axis at 90°W; ERA5 and GeoJSON start at 180°W.
  texture.repeat.set(1, 1)
  texture.offset.set(0.25, 0)
  texture.needsUpdate = true
}

function pruneTextureCache(keepUrl: string) {
  if (textureCache.size <= MAX_TEXTURE_CACHE_SIZE) return

  for (const [url, entry] of textureCache) {
    if (textureCache.size <= MAX_TEXTURE_CACHE_SIZE) break
    if (url === keepUrl || activeTextureUrls.has(url) || !entry.texture) continue
    entry.texture.dispose()
    textureCache.delete(url)
  }
}

function loadCoastlinePaths() {
  if (coastlinePathsPromise) return coastlinePathsPromise

  coastlinePathsPromise = fetch(resolvePublicAssetUrl(COASTLINES_URL))
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Coastline request failed with status ${response.status}.`)
      }
      return response.json() as Promise<unknown>
    })
    .then(convertGeoJsonToCoastlinePaths)
    .catch((error: unknown) => {
      coastlinePathsPromise = undefined
      throw error
    })

  return coastlinePathsPromise
}

export function GlobeCanvas({
  textureUrl,
  textureRevision,
  autoRotate,
  isFullscreen,
  resetSignal,
  onPointSelect,
  onTextureLoadingChange,
  onTextureError,
}: GlobeCanvasProps) {
  const { ref: containerRef, width, height } = useElementSize<HTMLDivElement>()
  const runtimeRef = useRef<Runtime | null>(null)
  const pointSelectRef = useRef(onPointSelect)
  const loadingChangeRef = useRef(onTextureLoadingChange)
  const textureErrorRef = useRef(onTextureError)

  pointSelectRef.current = onPointSelect
  loadingChangeRef.current = onTextureLoadingChange
  textureErrorRef.current = onTextureError

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100)
    setDefaultCameraPosition(camera)

    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    })
    renderer.setClearColor(0x000000, 0)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.domElement.setAttribute('role', 'img')
    renderer.domElement.setAttribute(
      'aria-label',
      'Интерактивный трёхмерный глобус с распределением выбранного поля ERA5',
    )
    renderer.domElement.style.display = 'block'
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.touchAction = 'none'
    container.appendChild(renderer.domElement)

    const sphereGeometry = new THREE.SphereGeometry(1, 384, 256)
    const materials: [THREE.MeshPhongMaterial, THREE.MeshPhongMaterial] = [
      createSphereMaterial(),
      createSphereMaterial(),
    ]
    const spheres: [THREE.Mesh, THREE.Mesh] = [
      new THREE.Mesh(sphereGeometry, materials[0]),
      new THREE.Mesh(sphereGeometry, materials[1]),
    ]
    spheres[1].visible = false
    scene.add(...spheres)

    const grid = createGeographicGrid()
    scene.add(grid)

    const atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.075, 96, 64),
      new THREE.MeshBasicMaterial({
        color: 0x93c5fd,
        transparent: true,
        opacity: 0.1,
        side: THREE.BackSide,
        depthWrite: false,
      }),
    )
    scene.add(atmosphere)

    scene.add(new THREE.HemisphereLight(0xffffff, 0x94a3b8, 1.65))
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.2)
    keyLight.position.set(-3, 2.5, 4)
    scene.add(keyLight)
    const fillLight = new THREE.DirectionalLight(0xbfdbfe, 0.8)
    fillLight.position.set(3, -1, -2)
    scene.add(fillLight)

    const marker = createPointMarker()
    scene.add(marker)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.055
    controls.enablePan = false
    controls.enableZoom = true
    controls.rotateSpeed = 0.55
    controls.zoomSpeed = 0.75
    controls.minDistance = 1.75
    controls.maxDistance = 4.25
    controls.autoRotateSpeed = 0.35
    controls.autoRotate = autoRotate
    controls.target.set(0, 0, 0)

    const runtime: Runtime = {
      renderer,
      scene,
      camera,
      controls,
      materials,
      spheres,
      marker,
      activeSphere: 0,
      autoRotateRequested: autoRotate,
      visible: true,
      disposed: false,
    }
    runtimeRef.current = runtime

    void loadLandMask()
      .then((mask) => {
        if (runtime.disposed) return

        materials.forEach((material) => {
          material.displacementMap = mask
          material.displacementScale = LAND_DISPLACEMENT_SCALE
          material.displacementBias = 0
          material.bumpMap = mask
          material.bumpScale = LAND_BUMP_SCALE
          material.emissiveMap = mask
          material.emissive.set(0xffffff)
          material.emissiveIntensity = 0.14
          material.needsUpdate = true
        })
      })
      .catch((caughtError: unknown) => {
        if (import.meta.env.DEV) {
          console.warn('Не удалось загрузить маску суши.', caughtError)
        }
      })

    let coastlineLayer: LineSegments2 | null = null
    void loadCoastlinePaths()
      .then((paths) => {
        const layer = createCoastlineLayer(paths)
        if (runtime.disposed) {
          disposeObject(layer)
          return
        }

        coastlineLayer = layer
        scene.add(layer)
      })
      .catch((caughtError: unknown) => {
        if (import.meta.env.DEV) {
          console.warn('Не удалось загрузить береговые линии глобуса.', caughtError)
        }
      })

    const pauseAutoRotate = () => {
      if (runtime.resumeTimer !== undefined) window.clearTimeout(runtime.resumeTimer)
      controls.autoRotate = false
    }
    const resumeAutoRotate = () => {
      if (runtime.resumeTimer !== undefined) window.clearTimeout(runtime.resumeTimer)
      runtime.resumeTimer = window.setTimeout(() => {
        controls.autoRotate = runtime.autoRotateRequested && !document.hidden
      }, 3000)
    }
    controls.addEventListener('start', pauseAutoRotate)
    controls.addEventListener('end', resumeAutoRotate)

    let pointerStart: { x: number; y: number } | null = null
    const handlePointerDown = (event: PointerEvent) => {
      pointerStart = { x: event.clientX, y: event.clientY }
    }
    const handlePointerUp = (event: PointerEvent) => {
      if (!pointerStart) return
      const distance = Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y)
      pointerStart = null
      if (distance > 5) return

      const coordinates = raycastCoordinates(event, camera, renderer.domElement, spheres)
      if (!coordinates) return

      positionMarker(marker, coordinates)
      pointSelectRef.current(coordinates)
    }
    renderer.domElement.addEventListener('pointerdown', handlePointerDown)
    renderer.domElement.addEventListener('pointerup', handlePointerUp)

    const handleDoubleClick = (event: MouseEvent) => event.preventDefault()
    renderer.domElement.addEventListener('dblclick', handleDoubleClick)

    const handleVisibilityChange = () => {
      controls.autoRotate = runtime.autoRotateRequested && !document.hidden
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)

    const observer = new IntersectionObserver(
      ([entry]) => {
        runtime.visible = entry.isIntersecting
      },
      { rootMargin: '120px' },
    )
    observer.observe(container)

    const render = () => {
      if (runtime.disposed) return
      if (runtime.visible && !document.hidden) {
        controls.update()
        renderer.render(scene, camera)
      }
      runtime.animationFrame = window.requestAnimationFrame(render)
    }
    render()

    return () => {
      runtime.disposed = true
      if (runtime.animationFrame !== undefined) window.cancelAnimationFrame(runtime.animationFrame)
      if (runtime.resumeTimer !== undefined) window.clearTimeout(runtime.resumeTimer)
      observer.disconnect()
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown)
      renderer.domElement.removeEventListener('pointerup', handlePointerUp)
      renderer.domElement.removeEventListener('dblclick', handleDoubleClick)
      controls.removeEventListener('start', pauseAutoRotate)
      controls.removeEventListener('end', resumeAutoRotate)
      controls.dispose()
      sphereGeometry.dispose()
      materials.forEach((material) => material.dispose())
      disposeObject(grid)
      if (coastlineLayer) {
        scene.remove(coastlineLayer)
        disposeObject(coastlineLayer)
      }
      disposeObject(atmosphere)
      disposeObject(marker)
      renderer.dispose()
      renderer.domElement.remove()
      if (runtime.currentTextureUrl) activeTextureUrls.delete(runtime.currentTextureUrl)
      runtimeRef.current = null
    }
  }, [containerRef])

  useEffect(() => {
    const runtime = runtimeRef.current
    if (!runtime || width <= 0 || height <= 0) return

    runtime.renderer.setSize(width, height, false)
    runtime.camera.aspect = width / height
    runtime.camera.fov = 38 / Math.min(1, runtime.camera.aspect)
    runtime.camera.updateProjectionMatrix()
  }, [height, width])

  useEffect(() => {
    const runtime = runtimeRef.current
    if (!runtime) return
    runtime.autoRotateRequested = autoRotate
    if (runtime.resumeTimer !== undefined) window.clearTimeout(runtime.resumeTimer)
    runtime.controls.autoRotate = autoRotate && !document.hidden
  }, [autoRotate])

  useEffect(() => {
    const runtime = runtimeRef.current
    if (!runtime || resetSignal === 0) return
    setDefaultCameraPosition(runtime.camera)
    runtime.controls.target.set(0, 0, 0)
    runtime.controls.update()
  }, [resetSignal])

  useEffect(() => {
    const runtime = runtimeRef.current
    if (!runtime) return

    if (!textureUrl) {
      if (runtime.currentTextureUrl) activeTextureUrls.delete(runtime.currentTextureUrl)
      runtime.currentTextureUrl = undefined
      runtime.spheres.forEach((sphere, index) => {
        const material = runtime.materials[index]
        material.map = null
        material.color.set(0x5b7f9c)
        material.opacity = index === 0 ? 1 : 0
        material.needsUpdate = true
        sphere.visible = index === 0
      })
      runtime.activeSphere = 0
      loadingChangeRef.current(false)
      textureErrorRef.current(null)
      return
    }

    let cancelled = false
    const previousUrl = runtime.currentTextureUrl
    activeTextureUrls.add(textureUrl)
    loadingChangeRef.current(true)
    textureErrorRef.current(null)

    loadTexture(textureUrl)
      .then((texture) => {
        if (cancelled || runtime.disposed) return

        const nextIndex = runtime.activeSphere === 0 ? 1 : 0
        const previousIndex = runtime.activeSphere
        const nextMaterial = runtime.materials[nextIndex]
        const previousMaterial = runtime.materials[previousIndex]
        const nextSphere = runtime.spheres[nextIndex]
        const previousSphere = runtime.spheres[previousIndex]

        nextMaterial.map = texture
        nextMaterial.color.set(0xffffff)
        nextMaterial.opacity = 0
        nextMaterial.needsUpdate = true
        nextSphere.visible = true

        const startedAt = performance.now()
        const duration = 220
        const fade = (now: number) => {
          if (cancelled || runtime.disposed) return
          const progress = Math.min(1, (now - startedAt) / duration)
          nextMaterial.opacity = progress
          previousMaterial.opacity = 1 - progress

          if (progress < 1) {
            window.requestAnimationFrame(fade)
            return
          }

          previousSphere.visible = false
          previousMaterial.opacity = 1
          runtime.activeSphere = nextIndex
          runtime.currentTextureUrl = textureUrl
          if (previousUrl && previousUrl !== textureUrl) activeTextureUrls.delete(previousUrl)
          loadingChangeRef.current(false)
          pruneTextureCache(textureUrl)
        }
        window.requestAnimationFrame(fade)
      })
      .catch((caughtError: unknown) => {
        if (cancelled) return
        activeTextureUrls.delete(textureUrl)
        loadingChangeRef.current(false)
        textureErrorRef.current(
          caughtError instanceof Error ? caughtError.message : 'Не удалось декодировать текстуру.',
        )
      })

    return () => {
      cancelled = true
      if (runtime.currentTextureUrl !== textureUrl) activeTextureUrls.delete(textureUrl)
    }
  }, [textureRevision, textureUrl])

  return (
    <div
      ref={containerRef}
      className={`relative min-w-0 overflow-hidden ${
        isFullscreen
          ? 'h-[62vh] min-h-[360px] max-h-[680px]'
          : 'h-[330px] sm:h-[400px] xl:h-[480px]'
      }`}
    >
      <div
        className="pointer-events-none absolute bottom-[7%] left-1/2 h-[8%] w-[46%] -translate-x-1/2 rounded-[50%] bg-slate-500/20 blur-xl"
        aria-hidden="true"
      />
      <div className="pointer-events-none absolute bottom-3 left-1/2 z-10 -translate-x-1/2 whitespace-nowrap rounded-full border border-slate-200 bg-white/90 px-3 py-1.5 text-[11px] font-medium text-slate-500 shadow-sm backdrop-blur-sm">
        <span className="sm:hidden">Вращение · Масштаб</span>
        <span className="hidden sm:inline">Перетащите, чтобы вращать · Колесо мыши — масштаб</span>
      </div>
    </div>
  )
}

function createSphereMaterial() {
  return new THREE.MeshPhongMaterial({
    color: 0x5b7f9c,
    shininess: 12,
    specular: 0x90b4cd,
    transparent: true,
    opacity: 1,
  })
}

function createGeographicGrid() {
  const group = new THREE.Group()
  const material = new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.18,
    depthWrite: false,
  })

  for (let latitude = -60; latitude <= 60; latitude += 30) {
    const points: THREE.Vector3[] = []
    const latitudeRadians = THREE.MathUtils.degToRad(latitude)
    const radius = Math.cos(latitudeRadians) * 1.003
    const y = Math.sin(latitudeRadians) * 1.003
    for (let longitude = -180; longitude <= 180; longitude += 3) {
      const radians = THREE.MathUtils.degToRad(longitude)
      points.push(new THREE.Vector3(radius * Math.sin(radians), y, radius * Math.cos(radians)))
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material))
  }

  for (let longitude = -150; longitude < 180; longitude += 30) {
    const points: THREE.Vector3[] = []
    const longitudeRadians = THREE.MathUtils.degToRad(longitude)
    for (let latitude = -90; latitude <= 90; latitude += 3) {
      const latitudeRadians = THREE.MathUtils.degToRad(latitude)
      const radius = Math.cos(latitudeRadians) * 1.003
      points.push(
        new THREE.Vector3(
          radius * Math.sin(longitudeRadians),
          Math.sin(latitudeRadians) * 1.003,
          radius * Math.cos(longitudeRadians),
        ),
      )
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material))
  }

  return group
}

function createCoastlineLayer(paths: CoastlinePath[]) {
  const positions: number[] = []

  for (const path of paths) {
    for (let index = 1; index < path.points.length; index += 1) {
      appendCoastlineArc(positions, path.points[index - 1], path.points[index])
    }
  }

  const geometry = new LineSegmentsGeometry()
  geometry.setPositions(positions)
  geometry.computeBoundingSphere()

  const material = new LineMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.9,
    alphaToCoverage: true,
    depthTest: true,
    depthWrite: false,
  })
  material.uniforms.linewidth.value = 2.4

  const coastlines = new LineSegments2(geometry, material)
  coastlines.name = 'coastlines'
  coastlines.renderOrder = 4
  return coastlines
}

function appendCoastlineArc(
  positions: number[],
  start: CoastlinePath['points'][number],
  end: CoastlinePath['points'][number],
) {
  const longitudeDelta = end.lng - start.lng
  const latitudeDelta = end.lat - start.lat
  const steps = Math.max(
    1,
    Math.ceil(Math.max(Math.abs(longitudeDelta), Math.abs(latitudeDelta)) / 1.5),
  )
  let previous = lonLatToSphere(start.lng, start.lat)

  for (let step = 1; step <= steps; step += 1) {
    const progress = step / steps
    const current = lonLatToSphere(
      start.lng + longitudeDelta * progress,
      start.lat + latitudeDelta * progress,
    )
    positions.push(previous.x, previous.y, previous.z, current.x, current.y, current.z)
    previous = current
  }
}

function lonLatToSphere(longitude: number, latitude: number) {
  const radius = COASTLINE_RADIUS
  const latitudeRadians = THREE.MathUtils.degToRad(THREE.MathUtils.clamp(latitude, -90, 90))
  const longitudeRadians = THREE.MathUtils.degToRad(longitude)
  const horizontalRadius = Math.cos(latitudeRadians) * radius

  return new THREE.Vector3(
    horizontalRadius * Math.sin(longitudeRadians),
    Math.sin(latitudeRadians) * radius,
    horizontalRadius * Math.cos(longitudeRadians),
  )
}

function createPointMarker() {
  const group = new THREE.Group()
  group.visible = false
  group.add(
    new THREE.Mesh(
      new THREE.SphereGeometry(0.025, 24, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    ),
  )
  group.add(
    new THREE.Mesh(
      new THREE.SphereGeometry(0.017, 24, 16),
      new THREE.MeshBasicMaterial({ color: 0x2563eb }),
    ),
  )
  return group
}

function positionMarker(marker: THREE.Group, coordinates: GlobeCoordinates) {
  const latitude = THREE.MathUtils.degToRad(coordinates.latitude)
  const longitude = THREE.MathUtils.degToRad(coordinates.longitude)
  const radius = MARKER_RADIUS
  marker.position.set(
    radius * Math.cos(latitude) * Math.sin(longitude),
    radius * Math.sin(latitude),
    radius * Math.cos(latitude) * Math.cos(longitude),
  )
  marker.visible = true
}

function setDefaultCameraPosition(camera: THREE.PerspectiveCamera) {
  const latitude = THREE.MathUtils.degToRad(24)
  const longitude = THREE.MathUtils.degToRad(12)
  const distance = 3.05
  camera.position.set(
    distance * Math.cos(latitude) * Math.sin(longitude),
    distance * Math.sin(latitude),
    distance * Math.cos(latitude) * Math.cos(longitude),
  )
  camera.lookAt(0, 0, 0)
}

function raycastCoordinates(
  event: PointerEvent,
  camera: THREE.PerspectiveCamera,
  canvas: HTMLCanvasElement,
  spheres: [THREE.Mesh, THREE.Mesh],
) {
  const rect = canvas.getBoundingClientRect()
  const pointer = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  )
  const raycaster = new THREE.Raycaster()
  raycaster.setFromCamera(pointer, camera)
  const intersection = raycaster.intersectObjects(
    spheres.filter((sphere) => sphere.visible),
    false,
  )[0]
  if (!intersection) return null

  const point = intersection.object.worldToLocal(intersection.point.clone()).normalize()
  return {
    latitude: THREE.MathUtils.radToDeg(Math.asin(point.y)),
    longitude: THREE.MathUtils.radToDeg(Math.atan2(point.x, point.z)),
  }
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
      child.geometry.dispose()
      const materials = Array.isArray(child.material) ? child.material : [child.material]
      materials.forEach((material) => material.dispose())
    }
  })
}

const landMaskUrl = resolvePublicAssetUrl('data/globe/land-mask.png')
let landMaskPromise: Promise<THREE.Texture> | undefined

function loadLandMask() {
  if (landMaskPromise) return landMaskPromise

  landMaskPromise = loadDataTexture(landMaskUrl).catch((error: unknown) => {
    landMaskPromise = undefined
    throw error
  })
  return landMaskPromise
}

function loadDataTexture(url: string): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    const loader = new THREE.TextureLoader()

    loader.load(
      url,
      (texture) => {
        texture.colorSpace = THREE.NoColorSpace
        configureEquirectangularTexture(texture)
        texture.minFilter = THREE.LinearMipmapLinearFilter
        texture.magFilter = THREE.LinearFilter

        resolve(texture)
      },
      undefined,
      (error) => {
        reject(
          error instanceof Error
            ? error
            : new Error('Не удалось загрузить land-mask.png'),
        )
      },
    )
  })
}

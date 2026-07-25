import { FileArchive, FileCheck2, UploadCloud, X } from 'lucide-react'
import { useRef, useState, type DragEvent } from 'react'
import { formatFileSize } from './fileValidation'

type CodecDropzoneProps = {
  file: File | null
  error: string | null
  disabled?: boolean
  onFileChange: (file: File | null) => void
}

export function CodecDropzone({
  file,
  error,
  disabled,
  onFileChange,
}: CodecDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const receiveDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    if (!disabled) onFileChange(event.dataTransfer.files[0] ?? null)
  }

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept=".npz,.npy,.nc,.zip"
        className="sr-only"
        disabled={disabled}
        onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
      />
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        onDragEnter={(event) => {
          event.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={receiveDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(event) => {
          if (!disabled && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault()
            inputRef.current?.click()
          }
        }}
        className={`codec-dropzone ui-focus-ring ${dragging ? 'codec-dropzone--active' : ''} ${
          error ? 'codec-dropzone--error' : ''
        }`}
      >
        {file ? (
          <>
            <span className="codec-dropzone__icon codec-dropzone__icon--ready">
              <FileCheck2 className="h-6 w-6" aria-hidden="true" />
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-[15px] font-bold text-[var(--text-primary)]">
                {file.name}
              </p>
              <p className="mt-1 text-[12px] text-[var(--text-muted)]">
                {formatFileSize(file.size)} · готов к проверке контракта
              </p>
            </div>
            <button
              type="button"
              aria-label="Убрать выбранный файл"
              disabled={disabled}
              onClick={(event) => {
                event.stopPropagation()
                onFileChange(null)
                if (inputRef.current) inputRef.current.value = ''
              }}
              className="ui-button-ghost ui-focus-ring inline-flex h-10 w-10 items-center justify-center rounded-xl"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </button>
          </>
        ) : (
          <>
            <span className="codec-dropzone__icon">
              {dragging ? (
                <FileArchive className="h-6 w-6" aria-hidden="true" />
              ) : (
                <UploadCloud className="h-6 w-6" aria-hidden="true" />
              )}
            </span>
            <div>
              <p className="text-[15px] font-bold text-[var(--text-primary)]">
                Перетащите ERA5-файл сюда
              </p>
              <p className="mt-1 text-[12px] leading-5 text-[var(--text-muted)]">
                NPZ, NPY, NetCDF или Zarr ZIP · один 28-канальный кадр · до 2 ГиБ
              </p>
            </div>
          </>
        )}
      </div>
      {error ? (
        <p className="mt-2 text-[12px] font-semibold text-rose-600" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  )
}

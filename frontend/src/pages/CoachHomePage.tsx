import { Users } from "lucide-react"

export function CoachHomePage() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
      <Users className="h-8 w-8" />
      <p className="text-sm">Select an athlete from the sidebar to view their calendar.</p>
    </div>
  )
}

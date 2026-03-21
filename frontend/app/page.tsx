"use client"

import { TopNav } from "@/components/terminal/top-nav"
import { LeftSidebar } from "@/components/terminal/left-sidebar"
import { RightSidebar } from "@/components/terminal/right-sidebar"
import { BottomPanel } from "@/components/terminal/bottom-panel"
import { ChartWorkspace } from "@/components/workspace/chart-workspace"
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable"

export default function TerminalPage() {
  return (
    <div className="h-screen w-screen flex flex-col bg-[#0B0F14] text-[#E6EDF3] overflow-hidden">
      {/* Top Nav */}
      <TopNav />

      {/* Main body: vertical split between workspace and bottom panel */}
      <ResizablePanelGroup direction="vertical" className="flex-1 min-h-0">
        {/* Upper section: sidebars + main chart area */}
        <ResizablePanel defaultSize={75} minSize={40}>
          <ResizablePanelGroup direction="horizontal" className="h-full">
            {/* Left sidebar */}
            <ResizablePanel defaultSize={15} minSize={10} maxSize={25}>
              <LeftSidebar />
            </ResizablePanel>

            <ResizableHandle className="bg-[#1E2A38] hover:bg-[#4CC9F0] transition-colors" />

            {/* Main chart workspace */}
            <ResizablePanel defaultSize={65} minSize={30}>
              <ChartWorkspace />
            </ResizablePanel>

            <ResizableHandle className="bg-[#1E2A38] hover:bg-[#4CC9F0] transition-colors" />

            {/* Right sidebar */}
            <ResizablePanel defaultSize={20} minSize={10} maxSize={30}>
              <RightSidebar />
            </ResizablePanel>
          </ResizablePanelGroup>
        </ResizablePanel>

        <ResizableHandle className="bg-[#1E2A38] hover:bg-[#4CC9F0] transition-colors" />

        {/* Bottom panel (logs) */}
        <ResizablePanel defaultSize={25} minSize={8} maxSize={50}>
          <BottomPanel />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}

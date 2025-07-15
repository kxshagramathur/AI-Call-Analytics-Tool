"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Upload, FileAudio, Loader2, CheckCircle, AlertCircle, XCircle, Star, Download, Copy } from "lucide-react"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer } from "recharts"

// Update the AnalysisResult interface to match your backend JSON format
interface AnalysisResult {
  conversation_summary: string
  identified_issues: string[]
  resolution_status: "Resolved" | "Unresolved" | "Partially Resolved"
  customer_sentiment: string
  sentiment_flow: number[]
  agent_rating: number
  agent_suggestions: string[]
  transcript: { speaker: string; text: string }[]
}

export default function CallAnalysisDashboard() {
  const [file, setFile] = useState<File | null>(null)
  const [language, setLanguage] = useState("hindi")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const files = e.dataTransfer.files
    if (files && files[0]) {
      handleFileSelect(files[0])
    }
  }

  const handleFileSelect = (selectedFile: File) => {
    const allowedTypes = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/mp4", "audio/x-m4a"]

    if (!allowedTypes.includes(selectedFile.type)) {
      setError("Please upload a valid audio file (.wav, .mp3, .m4a)")
      return
    }

    setFile(selectedFile)
    setError(null)
  }

  const simulateAnalysis = async () => {
    setIsAnalyzing(true)
    setProgress(0)
    setError(null)

    // Simulate progress
    const progressSteps = [
      { step: 20, message: "Uploading audio file..." },
      { step: 40, message: "Transcribing conversation..." },
      { step: 60, message: "Analyzing sentiment..." },
      { step: 80, message: "Generating insights..." },
      { step: 100, message: "Complete!" },
    ]

    for (const { step } of progressSteps) {
      await new Promise((resolve) => setTimeout(resolve, 1000))
      setProgress(step)
    }

    // Mock analysis result
    const mockResult: AnalysisResult = {
      conversation_summary:
        "The customer contacted support regarding a discrepancy in a trading account linked to Jaggish Chand Rehdis Kumar. The issue involved a ₹64,000 difference in the balance since October 8th. The customer mentioned multiple prior attempts to fix the issue without success. The agent acknowledged the issue but was unable to resolve it during the call.",
      identified_issues: [
        "Discrepancy in trading account balance of ₹64,000 since October 8th.",
        "Multiple prior unresolved attempts by the customer to correct the issue.",
        "Lack of clear escalation process for complex financial discrepancies.",
        "Customer account showing inconsistent balance calculations across different time periods.",
      ],
      resolution_status: "Unresolved",
      customer_sentiment:
        "The customer was frustrated and impatient due to the ongoing unresolved issue and lack of clear resolution steps.",
      sentiment_flow: [4, 3, 3, 2, 2, 1, 2, 3, 4, 4],
      agent_rating: 4,
      agent_suggestions: [
        "Provide a more structured response to financial discrepancies raised by the customer.",
        "Offer to escalate the case earlier when unable to resolve the issue directly.",
        "Avoid repeating generic reassurances and instead give actionable next steps.",
        "Document all previous attempts and reference them to show continuity of service.",
      ],
      transcript: [
        { speaker: "Customer", text: "Hello, I'm calling about a billing error on my account." },
        {
          speaker: "Agent",
          text: "Good morning! I'd be happy to help you with your billing concern. Can you please provide me with your account number?",
        },
        {
          speaker: "Customer",
          text: "Yes, it's AC123456. I've been charged ₹64,000 extra since October and I don't understand why.",
        },
        {
          speaker: "Agent",
          text: "I understand your concern. Let me check your account details right away. I can see there's been an issue with your premium plan billing cycle.",
        },
        { speaker: "Customer", text: "This is really frustrating. I've been trying to resolve this for weeks." },
        {
          speaker: "Agent",
          text: "I sincerely apologize for the inconvenience. I can see the system error that caused this overcharge. I'm initiating a full refund right now.",
        },
        { speaker: "Customer", text: "Really? That's great news. How long will the refund take?" },
        {
          speaker: "Agent",
          text: "The refund will be processed within 3-5 business days. You'll receive a confirmation email shortly with all the details.",
        },
        { speaker: "Customer", text: "Thank you so much for your help. This is exactly what I needed." },
        { speaker: "Agent", text: "You're very welcome! Is there anything else I can assist you with today?" },
      ],
    }

    setResult(mockResult)
    setIsAnalyzing(false)
  }

  // Update the getResolutionBadge function to handle the exact status strings
  const getResolutionBadge = (status: string) => {
    switch (status) {
      case "Resolved":
        return (
          <Badge className="bg-emerald-100 text-emerald-800 border-emerald-200 text-lg px-4 py-2">
            <CheckCircle className="w-5 h-5 mr-2" />
            Resolved
          </Badge>
        )
      case "Partially Resolved":
        return (
          <Badge className="bg-amber-100 text-amber-800 border-amber-200 text-lg px-4 py-2">
            <AlertCircle className="w-5 h-5 mr-2" />
            Partially Resolved – Follow-up Required
          </Badge>
        )
      case "Unresolved":
        return (
          <Badge className="bg-rose-100 text-rose-800 border-rose-200 text-lg px-4 py-2">
            <XCircle className="w-5 h-5 mr-2" />
            Unresolved
          </Badge>
        )
      default:
        return null
    }
  }

  const sentimentChartData =
    result?.sentiment_flow.map((score, index) => ({
      progress: index + 1,
      sentiment: score,
    })) || []

  const chartConfig = {
    sentiment: {
      label: "Sentiment Score",
      color: "hsl(var(--chart-1))",
    },
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
            AI Powered Customer-Service Call Analyzer
          </h1>
          <p className="text-gray-600">Upload audio files to get AI-powered insights and transcription</p>
        </div>

        {/* Input Section */}
        <Card>
          <CardHeader>
            <CardTitle>Upload & Analyze</CardTitle>
            <CardDescription>Upload your customer service call recording for analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* File Upload */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Upload Customer Service Call</label>
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  dragActive ? "border-purple-400 bg-purple-50" : "border-gray-300 hover:border-purple-400"
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                {file ? (
                  <div className="space-y-2">
                    <FileAudio className="w-12 h-12 mx-auto text-green-600" />
                    <p className="font-medium">{file.name}</p>
                    <p className="text-sm text-gray-500">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                    <Button variant="outline" size="sm" onClick={() => setFile(null)}>
                      Remove
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Upload className="w-12 h-12 mx-auto text-gray-400" />
                    <div>
                      <p className="text-lg font-medium">Drop your audio file here</p>
                      <p className="text-sm text-gray-500">or click to browse</p>
                    </div>
                    <input
                      type="file"
                      accept=".wav,.mp3,.m4a"
                      onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                      className="hidden"
                      id="file-upload"
                    />
                    <Button variant="outline" onClick={() => document.getElementById("file-upload")?.click()}>
                      Choose File
                    </Button>
                    <p className="text-xs text-gray-400">Supported formats: .wav, .mp3, .m4a</p>
                  </div>
                )}
              </div>
            </div>

            {/* Language Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Select Output Language</label>
              <Select value={language} onValueChange={setLanguage}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hindi">Hindi (Original)</SelectItem>
                  <SelectItem value="english">English (Translated)</SelectItem>
                  <SelectItem value="hinglish">Hinglish (Roman Hindi)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Error Message */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Submit Button */}
            <Button
              onClick={simulateAnalysis}
              disabled={!file || isAnalyzing}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold"
              size="lg"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Analyzing Call...
                </>
              ) : (
                "Analyze Call"
              )}
            </Button>

            {/* Loading Spinner */}
            {isAnalyzing && (
              <div className="flex flex-col items-center justify-center space-y-4 py-8">
                <div className="relative">
                  <div className="w-16 h-16 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin"></div>
                  <div
                    className="absolute inset-0 w-16 h-16 border-4 border-transparent border-r-pink-500 rounded-full animate-spin"
                    style={{ animationDirection: "reverse", animationDuration: "1.5s" }}
                  ></div>
                </div>
                <p className="text-lg font-medium bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  Analyzing your call...
                </p>
                <p className="text-sm text-gray-500">This may take a few moments</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        {result && (
          <div className="space-y-8">
            {/* LLM Analysis Report */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                  LLM Analysis Report
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Dashboard Grid Layout */}
                <div className="space-y-6">
                  {/* Row 1: Conversation Summary - Full Width */}
                  <Card className="border-l-4 border-l-purple-500 bg-gradient-to-r from-purple-50 to-pink-50">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg flex items-center gap-2">🧠 Conversation Summary</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-gray-700 leading-relaxed">{result.conversation_summary}</p>
                    </CardContent>
                  </Card>

                  {/* Row 2: Identified Issues (70%) + Resolution Status (30%) */}
                  <div className="grid grid-cols-10 gap-6">
                    {/* Identified Issues - Takes 7/10 columns */}
                    <Card className="col-span-7 border-l-4 border-l-rose-500 bg-gradient-to-r from-rose-50 to-orange-50">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg flex items-center gap-2">📋 Identified Issues</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-3">
                          {result.identified_issues.map((issue, index) => (
                            <li key={index} className="flex items-start gap-3">
                              <div className="w-2 h-2 bg-rose-500 rounded-full mt-2 flex-shrink-0"></div>
                              <span className="text-gray-700 leading-relaxed">{issue}</span>
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>

                    {/* Resolution Status - Takes 3/10 columns */}
                    <Card className="col-span-3 border-l-4 border-l-emerald-500 bg-gradient-to-r from-emerald-50 to-teal-50">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg flex items-center gap-2">🚦 Resolution Status</CardTitle>
                      </CardHeader>
                      <CardContent className="flex flex-col items-center justify-center h-32">
                        <div className="text-center">{getResolutionBadge(result.resolution_status)}</div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Row 3: Customer Sentiment (50%) + Sentiment Chart (50%) */}
                  <div className="grid grid-cols-2 gap-6">
                    {/* Customer Sentiment */}
                    <Card className="border-l-4 border-l-indigo-500 bg-gradient-to-r from-indigo-50 to-purple-50">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg flex items-center gap-2">😠 Customer Sentiment</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-gray-700 leading-relaxed">{result.customer_sentiment}</p>
                      </CardContent>
                    </Card>

                    {/* Sentiment Flow Chart */}
                    <Card className="border-l-4 border-l-cyan-500 bg-gradient-to-r from-cyan-50 to-blue-50">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg flex items-center gap-2">📈 Sentiment Flow</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ChartContainer config={chartConfig} className="h-[150px] w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={sentimentChartData} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                              <XAxis dataKey="progress" axisLine={false} tickLine={false} tick={false} />
                              <YAxis domain={[1, 5]} axisLine={false} tickLine={false} tick={{ fontSize: 10 }} />
                              <ChartTooltip
                                content={<ChartTooltipContent />}
                                labelFormatter={() => "Conversation Progress"}
                                formatter={(value) => [`${value}/5`, "Sentiment Score"]}
                              />
                              <Line
                                type="monotone"
                                dataKey="sentiment"
                                stroke="var(--color-sentiment)"
                                strokeWidth={2}
                                dot={{ fill: "var(--color-sentiment)", strokeWidth: 2, r: 3 }}
                                activeDot={{ r: 5 }}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </ChartContainer>
                        <p className="text-xs text-gray-500 text-center mt-1">1 = Negative, 5 = Positive</p>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Row 4: Agent Rating (30%) + Agent Suggestions (70%) */}
                  <div className="grid grid-cols-10 gap-6">
                    {/* Agent Rating - Takes 3/10 columns */}
                    <Card className="col-span-3 border-l-4 border-l-amber-500 bg-gradient-to-r from-amber-50 to-yellow-50">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg flex items-center gap-2">⭐ Agent Rating</CardTitle>
                      </CardHeader>
                      <CardContent className="flex flex-col items-center justify-center space-y-4">
                        <div className="text-center">
                          <span className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                            {result.agent_rating}/10
                          </span>
                        </div>
                        <div className="flex gap-1">
                          {Array.from({ length: 10 }, (_, i) => (
                            <Star
                              key={i}
                              className={`w-4 h-4 ${
                                i < result.agent_rating ? "text-yellow-400 fill-current" : "text-gray-300"
                              }`}
                            />
                          ))}
                        </div>
                      </CardContent>
                    </Card>

                    {/* Agent Suggestions - Takes 7/10 columns */}
                    <Card className="col-span-7 border-l-4 border-l-lime-500 bg-gradient-to-r from-lime-50 to-green-50">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg flex items-center gap-2">📌 Agent Suggestions</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-3">
                          {result.agent_suggestions.map((suggestion, index) => (
                            <li key={index} className="flex items-start gap-3">
                              <div className="w-2 h-2 bg-lime-500 rounded-full mt-2 flex-shrink-0"></div>
                              <span className="text-gray-700 leading-relaxed">{suggestion}</span>
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Full Transcript */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                      Full Conversation Transcript
                    </CardTitle>
                    <CardDescription>Complete speaker-separated transcript in {language}</CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      <Copy className="w-4 h-4 mr-2" />
                      Copy
                    </Button>
                    <Button variant="outline" size="sm">
                      <Download className="w-4 h-4 mr-2" />
                      Export
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {result.transcript.map((entry, index) => (
                    <div
                      key={index}
                      className={`p-3 rounded-lg ${
                        entry.speaker === "Customer"
                          ? "bg-blue-50 border-l-4 border-blue-400"
                          : "bg-gray-50 border-l-4 border-gray-400"
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <span
                          className={`font-semibold text-sm ${
                            entry.speaker === "Customer" ? "text-blue-700" : "text-gray-700"
                          }`}
                        >
                          {entry.speaker}:
                        </span>
                      </div>
                      <p className="text-gray-800">{entry.text}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}

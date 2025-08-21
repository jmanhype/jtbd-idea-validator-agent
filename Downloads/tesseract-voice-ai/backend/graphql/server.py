"""
GraphQL Server with Strawberry and FastAPI
Provides test control plane operations with dry-run support
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Optional, AsyncGenerator
from enum import Enum

import strawberry
from strawberry.asgi import GraphQL
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Enums
@strawberry.enum
class TestStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"

@strawberry.enum
class FileAction(Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"

@strawberry.enum
class ProgressType(Enum):
    TEST_START = "TEST_START"
    TEST_COMPLETE = "TEST_COMPLETE"
    TEST_PROGRESS = "TEST_PROGRESS"
    SCAFFOLD_START = "SCAFFOLD_START"
    SCAFFOLD_COMPLETE = "SCAFFOLD_COMPLETE"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"

# Input Types
@strawberry.input
class TestOptions:
    parallel: Optional[bool] = False
    timeout: Optional[int] = 30000
    verbose: Optional[bool] = False
    coverage: Optional[bool] = False

@strawberry.input
class ScaffoldOptions:
    typescript: Optional[bool] = True
    css_module: Optional[bool] = True
    tests: Optional[bool] = True
    storybook: Optional[bool] = False

@strawberry.input
class ModificationInput:
    file_index: int
    new_content: str

@strawberry.input
class ConfirmationInput:
    confirmed: bool
    reason: Optional[str] = None
    modifications: Optional[List[ModificationInput]] = None

# Output Types
@strawberry.type
class TestResult:
    name: str
    status: TestStatus
    duration: float
    error: Optional[str] = None
    assertions: int = 0

@strawberry.type
class TestRun:
    id: str
    suite: str
    status: TestStatus
    start_time: str
    end_time: Optional[str] = None
    results: List[TestResult]
    dry_run: bool

@strawberry.type
class TestSuite:
    name: str
    description: str
    test_count: int
    estimated_duration: float

@strawberry.type
class FileChange:
    path: str
    action: FileAction
    content: Optional[str] = None
    diff: Optional[str] = None

@strawberry.type
class ScaffoldResult:
    id: str
    template: str
    files: List[FileChange]
    dry_run: bool
    preview: str

@strawberry.type
class TemplateVariable:
    name: str
    type: str
    required: bool
    default: Optional[str] = None

@strawberry.type
class ScaffoldTemplate:
    name: str
    description: str
    category: str
    variables: List[TemplateVariable]

@strawberry.type
class ProgressUpdate:
    run_id: str
    type: ProgressType
    progress: float
    message: str
    details: Optional[str] = None
    timestamp: str

@strawberry.type
class SystemEvent:
    id: str
    type: str
    payload: str
    timestamp: str

@strawberry.type
class ApplyResult:
    success: bool
    message: str
    applied_changes: List[FileChange]
    errors: List[str]

# In-memory storage for demo
class TestStore:
    def __init__(self):
        self.test_runs = {}
        self.scaffold_results = {}
        self.progress_queue = asyncio.Queue()
        self.event_queue = asyncio.Queue()
    
    def add_test_run(self, test_run: TestRun):
        self.test_runs[test_run.id] = test_run
    
    def get_test_run(self, run_id: str) -> Optional[TestRun]:
        return self.test_runs.get(run_id)
    
    def add_scaffold_result(self, result: ScaffoldResult):
        self.scaffold_results[result.id] = result
    
    def get_scaffold_result(self, result_id: str) -> Optional[ScaffoldResult]:
        return self.scaffold_results.get(result_id)
    
    async def emit_progress(self, update: ProgressUpdate):
        await self.progress_queue.put(update)
    
    async def emit_event(self, event: SystemEvent):
        await self.event_queue.put(event)

store = TestStore()

# Query implementation
@strawberry.type
class Query:
    @strawberry.field
    def test_status(self, run_id: str) -> TestRun:
        run = store.get_test_run(run_id)
        if not run:
            raise ValueError(f"Test run {run_id} not found")
        return run
    
    @strawberry.field
    def available_tests(self) -> List[TestSuite]:
        return [
            TestSuite(
                name="unit",
                description="Unit tests for components",
                test_count=42,
                estimated_duration=12.5
            ),
            TestSuite(
                name="integration",
                description="Integration tests for API",
                test_count=18,
                estimated_duration=25.3
            ),
            TestSuite(
                name="e2e",
                description="End-to-end browser tests",
                test_count=8,
                estimated_duration=45.0
            )
        ]
    
    @strawberry.field
    def scaffold_templates(self) -> List[ScaffoldTemplate]:
        return [
            ScaffoldTemplate(
                name="react-component",
                description="React component with TypeScript",
                category="component",
                variables=[
                    TemplateVariable(
                        name="componentName",
                        type="string",
                        required=True,
                        default=None
                    ),
                    TemplateVariable(
                        name="hasState",
                        type="boolean",
                        required=False,
                        default="false"
                    )
                ]
            ),
            ScaffoldTemplate(
                name="api-route",
                description="Next.js API route",
                category="api",
                variables=[
                    TemplateVariable(
                        name="routeName",
                        type="string",
                        required=True,
                        default=None
                    ),
                    TemplateVariable(
                        name="method",
                        type="string",
                        required=False,
                        default="GET"
                    )
                ]
            )
        ]

# Mutation implementation
@strawberry.type
class Mutation:
    @strawberry.mutation
    async def run_tests(
        self,
        suite: str,
        dry_run: bool = True,
        options: Optional[TestOptions] = None
    ) -> TestRun:
        run_id = str(uuid.uuid4())
        test_run = TestRun(
            id=run_id,
            suite=suite,
            status=TestStatus.PENDING if dry_run else TestStatus.RUNNING,
            start_time=datetime.now().isoformat(),
            end_time=None,
            results=[],
            dry_run=dry_run
        )
        
        store.add_test_run(test_run)
        
        # Simulate test execution
        if not dry_run:
            asyncio.create_task(self._execute_tests(test_run))
        else:
            # In dry-run, just show what would be executed
            test_run.results = [
                TestResult(
                    name=f"test_{i}",
                    status=TestStatus.PENDING,
                    duration=0.0,
                    assertions=0
                )
                for i in range(5)
            ]
        
        return test_run
    
    @strawberry.mutation
    async def scaffold_page(
        self,
        template: str,
        name: str,
        path: str,
        dry_run: bool = True,
        options: Optional[ScaffoldOptions] = None
    ) -> ScaffoldResult:
        result_id = str(uuid.uuid4())
        
        # Generate file changes based on template
        files = []
        if template == "react-component":
            component_content = f"""import React from 'react';

interface {name}Props {{
  // Define props here
}}

export const {name}: React.FC<{name}Props> = (props) => {{
  return (
    <div>
      <h1>{name}</h1>
    </div>
  );
}};

export default {name};"""
            
            files.append(FileChange(
                path=f"{path}/{name}.tsx",
                action=FileAction.CREATE,
                content=component_content,
                diff=None
            ))
            
            if options and options.tests:
                test_content = f"""import {{ render, screen }} from '@testing-library/react';
import {name} from './{name}';

describe('{name}', () => {{
  it('renders correctly', () => {{
    render(<{name} />);
    expect(screen.getByText('{name}')).toBeInTheDocument();
  }});
}});"""
                
                files.append(FileChange(
                    path=f"{path}/{name}.test.tsx",
                    action=FileAction.CREATE,
                    content=test_content,
                    diff=None
                ))
        
        result = ScaffoldResult(
            id=result_id,
            template=template,
            files=files,
            dry_run=dry_run,
            preview=f"Will create {len(files)} files for {name}"
        )
        
        store.add_scaffold_result(result)
        return result
    
    @strawberry.mutation
    async def apply_changes(
        self,
        run_id: str,
        confirmation: ConfirmationInput
    ) -> ApplyResult:
        if not confirmation.confirmed:
            return ApplyResult(
                success=False,
                message=f"Changes cancelled: {confirmation.reason or 'User cancelled'}",
                applied_changes=[],
                errors=[]
            )
        
        # Apply changes (in real implementation, would actually modify files)
        scaffold_result = store.get_scaffold_result(run_id)
        if scaffold_result:
            # Simulate applying changes
            return ApplyResult(
                success=True,
                message=f"Successfully applied {len(scaffold_result.files)} changes",
                applied_changes=scaffold_result.files,
                errors=[]
            )
        
        test_run = store.get_test_run(run_id)
        if test_run:
            # Execute actual tests
            test_run.status = TestStatus.RUNNING
            asyncio.create_task(self._execute_tests(test_run))
            return ApplyResult(
                success=True,
                message=f"Test run {run_id} started",
                applied_changes=[],
                errors=[]
            )
        
        return ApplyResult(
            success=False,
            message=f"Run ID {run_id} not found",
            applied_changes=[],
            errors=[f"Invalid run ID: {run_id}"]
        )
    
    async def _execute_tests(self, test_run: TestRun):
        """Simulate test execution with progress updates"""
        total_tests = 5
        for i in range(total_tests):
            await asyncio.sleep(1)  # Simulate test execution time
            
            # Update progress
            progress = (i + 1) / total_tests * 100
            await store.emit_progress(ProgressUpdate(
                run_id=test_run.id,
                type=ProgressType.TEST_PROGRESS,
                progress=progress,
                message=f"Running test {i + 1} of {total_tests}",
                details=None,
                timestamp=datetime.now().isoformat()
            ))
            
            # Add test result
            test_run.results.append(TestResult(
                name=f"test_{i}",
                status=TestStatus.PASSED if i % 3 != 0 else TestStatus.FAILED,
                duration=1.2 + i * 0.3,
                error=None if i % 3 != 0 else "Assertion failed",
                assertions=3
            ))
        
        test_run.status = TestStatus.PASSED if all(
            r.status == TestStatus.PASSED for r in test_run.results
        ) else TestStatus.FAILED
        test_run.end_time = datetime.now().isoformat()

# Subscription implementation
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def run_progress(self, run_id: str) -> AsyncGenerator[ProgressUpdate, None]:
        """Subscribe to progress updates for a specific run"""
        while True:
            try:
                update = await asyncio.wait_for(store.progress_queue.get(), timeout=1.0)
                if update.run_id == run_id:
                    yield update
            except asyncio.TimeoutError:
                continue
    
    @strawberry.subscription
    async def system_events(self) -> AsyncGenerator[SystemEvent, None]:
        """Subscribe to all system events"""
        while True:
            try:
                event = await asyncio.wait_for(store.event_queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                continue

# Create GraphQL schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)

# Create FastAPI app
app = FastAPI(title="Test Control Plane GraphQL API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GraphQL endpoint
graphql_app = GraphQL(
    schema,
    subscription_protocols=[GRAPHQL_TRANSPORT_WS_PROTOCOL]
)

app.mount("/graphql", graphql_app)

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "graphql-server"}

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=4000,
        reload=True,
        log_level="info"
    )
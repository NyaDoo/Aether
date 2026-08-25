<template>
  <div class="space-y-5 pb-8">
    <PageHeader
      title="素材库"
      :description="isAdmin ? '查看并管理所有用户的方舟素材与真人验证状态' : '录入、分组并复用视频生成素材'"
      :icon="Library"
    >
      <template #actions>
        <Button
          variant="outline"
          :disabled="!canCreate || verificationSessionPending || hasPendingVerificationSession"
          :title="createActionTitle('创建火山方舟真人验证会话')"
          @click="startVerification"
        >
          <Loader2
            v-if="verificationSessionPending"
            class="mr-2 h-4 w-4 animate-spin"
          />
          <UserRoundCheck
            v-else
            class="mr-2 h-4 w-4"
          />
          真人验证
        </Button>
        <Button
          variant="outline"
          :disabled="!canCreate || creatingFromUrl || creatableGroups.length === 0"
          :title="createActionTitle(creatableGroups.length === 0 ? '请先创建 AIGC 素材组' : '通过公开 URL 创建素材')"
          @click="openUrlUploadDialog"
        >
          <Link2 class="mr-2 h-4 w-4" />
          从 URL 创建
        </Button>
      </template>
    </PageHeader>

    <Card
      v-if="activeVerificationSession"
      class="overflow-hidden border-amber-500/30"
    >
      <div class="flex flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:px-5">
        <div class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-amber-500/10">
          <Loader2
            v-if="verificationSessionIsPending"
            class="h-4 w-4 animate-spin text-amber-600 dark:text-amber-300"
          />
          <UserRoundCheck
            v-else
            class="h-4 w-4 text-amber-600 dark:text-amber-300"
          />
        </div>
        <div class="min-w-0 flex-1">
          <p class="text-sm font-medium">
            {{ verificationSessionTitle }}
          </p>
          <p class="mt-0.5 text-xs leading-5 text-muted-foreground">
            {{ verificationSessionDescription }}
          </p>
        </div>
        <Button
          v-if="verificationSessionUrl && verificationSessionIsPending"
          variant="outline"
          size="sm"
          class="shrink-0"
          @click="reopenVerificationSession"
        >
          <ExternalLink class="mr-2 h-4 w-4" />
          打开验证页
        </Button>
      </div>
    </Card>

    <div class="grid items-start gap-4 xl:grid-cols-[250px_minmax(0,1fr)]">
      <Card class="overflow-hidden xl:sticky xl:top-20">
        <div class="flex items-center justify-between border-b border-border/60 px-4 py-3">
          <div>
            <h2 class="text-sm font-semibold">
              素材组
            </h2>
            <p class="mt-0.5 text-[11px] text-muted-foreground">
              AIGC 与真人验证组
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            class="h-8 w-8 rounded-lg"
            aria-label="创建素材组"
            :disabled="!canMutate"
            :title="createGroupActionTitle()"
            @click="openCreateGroupDialog"
          >
            <FolderPlus class="h-4 w-4" />
          </Button>
        </div>

        <div class="p-2">
          <button
            type="button"
            class="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors"
            :class="selectedGroupId === '' ? 'bg-primary/10 text-primary' : 'text-foreground hover:bg-muted/60'"
            @click="selectGroup('')"
          >
            <Library class="h-4 w-4 shrink-0" />
            <span class="min-w-0 flex-1 truncate font-medium">全部素材</span>
            <span class="text-xs tabular-nums text-muted-foreground">{{ total }}</span>
          </button>

          <div
            v-if="groupsLoading"
            class="space-y-2 px-2 py-3"
          >
            <Skeleton class="h-9 w-full" />
            <Skeleton class="h-9 w-full" />
          </div>
          <div
            v-else-if="groups.length === 0"
            class="px-3 py-8 text-center"
          >
            <Folder class="mx-auto h-7 w-7 text-muted-foreground/50" />
            <p class="mt-2 text-xs text-muted-foreground">
              尚未创建素材组
            </p>
          </div>
          <div
            v-else
            class="mt-1 space-y-1"
          >
            <div
              v-for="group in groups"
              :key="group.id"
              class="group/row flex w-full items-center rounded-lg transition-colors"
              :class="selectedGroupId === group.id ? 'bg-primary/10 text-primary' : 'text-foreground hover:bg-muted/60'"
            >
              <button
                type="button"
                class="flex min-w-0 flex-1 items-center gap-2 px-3 py-2 text-left text-sm"
                @click="selectGroup(group.id)"
              >
                <FolderOpen
                  v-if="selectedGroupId === group.id"
                  class="h-4 w-4 shrink-0"
                />
                <Folder
                  v-else
                  class="h-4 w-4 shrink-0 text-muted-foreground"
                />
                <span class="min-w-0 flex-1">
                  <span class="block truncate">{{ group.name }}</span>
                  <span class="mt-0.5 block truncate font-mono text-[9px] text-muted-foreground">{{ group.id }}</span>
                </span>
                <span
                  v-if="group.group_type === 'LivenessFace'"
                  class="rounded bg-amber-500/10 px-1 py-0.5 text-[9px] font-medium text-amber-700 dark:text-amber-300"
                >真人</span>
                <span class="text-xs tabular-nums text-muted-foreground">{{ group.asset_count }}</span>
              </button>
              <DropdownMenu>
                <DropdownMenuTrigger as-child>
                  <Button
                    variant="ghost"
                    size="icon"
                    class="mr-1 h-7 w-7 shrink-0 rounded-md opacity-100 transition-opacity sm:opacity-0 sm:focus:opacity-100 sm:group-hover/row:opacity-100"
                    :aria-label="`管理素材组 ${group.name}`"
                    :title="`管理素材组 ${group.name}`"
                  >
                    <MoreHorizontal class="h-3.5 w-3.5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  align="end"
                  class="w-40"
                >
                  <DropdownMenuItem
                    :disabled="!canMutate"
                    @select="openRenameGroupDialog(group)"
                  >
                    <Pencil class="mr-2 h-4 w-4" />
                    重命名素材组
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    class="text-destructive focus:text-destructive"
                    :disabled="!canMutate || deletingGroupId === group.id"
                    @select="deleteGroup(group)"
                  >
                    <Trash2 class="mr-2 h-4 w-4" />
                    {{ deletingGroupId === group.id ? '删除中...' : '删除素材组' }}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </div>
      </Card>

      <Card class="min-w-0 overflow-hidden">
        <div class="border-b border-border/60 px-4 py-4 sm:px-5">
          <div class="flex flex-col gap-3 2xl:flex-row 2xl:items-center 2xl:justify-between">
            <div class="min-w-0">
              <div class="flex items-center gap-2">
                <h2 class="truncate text-base font-semibold">
                  {{ activeGroupName }}
                </h2>
                <Badge variant="secondary">
                  {{ total }} 项
                </Badge>
              </div>
              <div class="mt-1 flex flex-wrap gap-3 text-xs text-muted-foreground">
                <span>Active {{ pageStatusCounts.active }}</span>
                <span>Processing {{ pageStatusCounts.processing }}</span>
                <span>Failed {{ pageStatusCounts.failed }}</span>
              </div>
            </div>

            <div class="flex flex-wrap items-center gap-2">
              <div class="relative min-w-[190px] flex-1 2xl:w-56 2xl:flex-none">
                <Search class="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  v-model="searchQuery"
                  class="h-9 pl-9 text-sm"
                  placeholder="搜索名称或素材 ID"
                  aria-label="搜索名称或素材 ID"
                  @keyup.enter="applySearchNow"
                />
              </div>

              <Select v-model="filterType">
                <SelectTrigger class="h-9 w-28 text-xs">
                  <SelectValue placeholder="类型" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">
                    全部类型
                  </SelectItem>
                  <SelectItem value="image">
                    图片
                  </SelectItem>
                  <SelectItem value="video">
                    视频
                  </SelectItem>
                  <SelectItem value="audio">
                    音频
                  </SelectItem>
                </SelectContent>
              </Select>

              <Select v-model="filterStatus">
                <SelectTrigger class="h-9 w-32 text-xs">
                  <SelectValue placeholder="状态" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">
                    全部状态
                  </SelectItem>
                  <SelectItem value="Processing">
                    Processing
                  </SelectItem>
                  <SelectItem value="Active">
                    Active
                  </SelectItem>
                  <SelectItem value="Failed">
                    Failed
                  </SelectItem>
                </SelectContent>
              </Select>

              <Input
                v-if="isAdmin"
                v-model="adminUserId"
                class="h-9 w-36 text-xs"
                placeholder="用户 ID"
                aria-label="按用户 ID 筛选"
                @keyup.enter="applyAdminUserFilter"
              />
              <Button
                v-if="isAdmin"
                variant="outline"
                size="sm"
                class="h-9"
                :disabled="!hasUnappliedAdminOwner"
                @click="applyAdminUserFilter"
              >
                应用用户
              </Button>

              <div class="flex rounded-lg border border-border/60 bg-muted/20 p-0.5">
                <Button
                  variant="ghost"
                  size="icon"
                  class="h-7 w-7 rounded-md"
                  :class="viewMode === 'grid' ? 'bg-background text-primary shadow-sm' : ''"
                  title="网格视图"
                  aria-label="网格视图"
                  :aria-pressed="viewMode === 'grid'"
                  @click="viewMode = 'grid'"
                >
                  <Grid2X2 class="h-3.5 w-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  class="h-7 w-7 rounded-md"
                  :class="viewMode === 'list' ? 'bg-background text-primary shadow-sm' : ''"
                  title="列表视图"
                  aria-label="列表视图"
                  :aria-pressed="viewMode === 'list'"
                  @click="viewMode = 'list'"
                >
                  <List class="h-3.5 w-3.5" />
                </Button>
              </div>

              <RefreshButton
                :loading="loading"
                title="刷新素材库"
                @click="refresh"
              />
            </div>
          </div>
        </div>

        <LoadingState
          v-if="loading && assets.length === 0"
          variant="skeleton"
          size="lg"
        />
        <EmptyState
          v-else-if="loadError"
          type="error"
          title="素材加载失败"
          :description="loadError"
          action-text="重新加载"
          @action="refresh"
        />
        <EmptyState
          v-else-if="assets.length === 0"
          :type="hasActiveFilters ? 'filter' : 'empty'"
          :icon="Library"
          :title="hasActiveFilters ? '没有匹配的素材' : '素材库还是空的'"
          :description="emptyStateDescription"
          :action-text="emptyActionText"
          :action-icon="creatableGroups.length === 0 ? FolderPlus : Link2"
          @action="handleEmptyAction"
        />

        <div
          v-else-if="viewMode === 'grid'"
          class="grid gap-4 p-4 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4"
        >
          <article
            v-for="asset in assets"
            :key="asset.id"
            class="group overflow-hidden rounded-lg border border-border/60 bg-card transition-colors hover:border-primary/40"
          >
            <button
              type="button"
              class="relative flex aspect-video w-full items-center justify-center overflow-hidden bg-muted/40 text-left"
              :disabled="normalizeMaterialAssetStatus(asset.status) !== 'active'"
              :aria-label="`预览素材 ${asset.name}`"
              @click="openPreview(asset)"
            >
              <MaterialAssetThumbnail
                v-if="normalizeMaterialAssetStatus(asset.status) === 'active' && normalizedMediaType(asset) === 'image'"
                :asset="asset"
                :scope="scope"
                :owner-user-id="ownerUserId"
              />
              <component
                :is="mediaIcon(normalizedMediaType(asset))"
                v-else-if="normalizeMaterialAssetStatus(asset.status) === 'active'"
                class="h-10 w-10 text-muted-foreground/55"
              />
              <div
                v-else-if="normalizeMaterialAssetStatus(asset.status) === 'failed'"
                class="px-5 text-center"
              >
                <AlertCircle class="mx-auto h-8 w-8 text-destructive" />
                <p class="mt-2 line-clamp-2 text-xs text-destructive">
                  {{ materialAssetErrorMessage(asset) }}
                </p>
              </div>
              <div
                v-else
                class="text-center"
              >
                <Loader2 class="mx-auto h-7 w-7 animate-spin text-primary" />
                <p class="mt-2 text-xs text-muted-foreground">
                  正在处理素材
                </p>
              </div>

              <div class="absolute left-2 top-2 flex flex-wrap gap-1.5">
                <Badge
                  variant="outline"
                  :class="assetStatusClass(asset.status)"
                >
                  {{ materialAssetStatusLabel(asset.status) }}
                </Badge>
                <Badge
                  v-if="materialAssetRequiresVerification(asset)"
                  variant="warning"
                >
                  真人验证
                </Badge>
              </div>
              <span
                v-if="normalizeMaterialAssetStatus(asset.status) === 'active'"
                class="absolute inset-0 flex items-center justify-center bg-black/0 opacity-0 transition-all group-hover:bg-black/25 group-hover:opacity-100"
              >
                <Eye class="h-6 w-6 text-white" />
              </span>
            </button>

            <div class="p-3">
              <div class="flex items-start gap-2">
                <div class="min-w-0 flex-1">
                  <h3
                    class="truncate text-sm font-semibold"
                    :title="asset.name"
                  >
                    {{ asset.name }}
                  </h3>
                  <p class="mt-1 truncate font-mono text-[11px] text-muted-foreground">
                    {{ asset.id }}
                  </p>
                  <p
                    v-if="officialAssetUrl(asset)"
                    class="mt-0.5 truncate font-mono text-[10px] text-muted-foreground/80"
                    :title="officialAssetUrl(asset) || undefined"
                  >
                    {{ officialAssetUrl(asset) }}
                  </p>
                </div>
                <MaterialAssetActionsMenu
                  :asset="asset"
                  :can-mutate="canMutate"
                  :deleting="deletingAssetId === asset.id"
                  @preview="openPreview"
                  @copy="copyAssetUri"
                  @use-for-video="useForVideo"
                  @rename="openRenameDialog"
                  @delete="deleteAsset"
                />
              </div>
              <div class="mt-3 flex items-center justify-between gap-3 text-[11px] text-muted-foreground">
                <span class="truncate">{{ asset.group_name || '素材组未知' }}</span>
                <span class="shrink-0">{{ assetMetadata(asset) }}</span>
              </div>
              <div
                v-if="isAdmin && (asset.username || asset.user_id)"
                class="mt-2 flex items-center gap-1 text-[11px] text-muted-foreground"
              >
                <User class="h-3 w-3" />
                <span class="truncate">{{ asset.username || asset.user_id }}</span>
              </div>
            </div>
          </article>
        </div>

        <div
          v-else
          class="overflow-x-auto"
        >
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>素材</TableHead>
                <TableHead>素材组</TableHead>
                <TableHead>状态</TableHead>
                <TableHead v-if="isAdmin">
                  用户 / Provider
                </TableHead>
                <TableHead>媒体信息</TableHead>
                <TableHead>创建时间</TableHead>
                <TableHead class="w-16 text-right">
                  操作
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow
                v-for="asset in assets"
                :key="asset.id"
              >
                <TableCell>
                  <div class="flex min-w-[220px] items-center gap-3">
                    <button
                      type="button"
                      class="flex h-10 w-14 shrink-0 items-center justify-center overflow-hidden rounded-md bg-muted/50"
                      :disabled="normalizeMaterialAssetStatus(asset.status) !== 'active'"
                      @click="openPreview(asset)"
                    >
                      <MaterialAssetThumbnail
                        v-if="normalizeMaterialAssetStatus(asset.status) === 'active' && normalizedMediaType(asset) === 'image'"
                        :asset="asset"
                        :scope="scope"
                        :owner-user-id="ownerUserId"
                      />
                      <component
                        :is="mediaIcon(normalizedMediaType(asset))"
                        v-else
                        class="h-4 w-4 text-muted-foreground"
                      />
                    </button>
                    <div class="min-w-0">
                      <p
                        class="truncate text-sm font-medium"
                        :title="asset.name"
                      >
                        {{ asset.name }}
                      </p>
                      <p class="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                        {{ asset.id }}
                      </p>
                      <p
                        v-if="officialAssetUrl(asset)"
                        class="mt-0.5 max-w-[360px] truncate font-mono text-[10px] text-muted-foreground/80"
                        :title="officialAssetUrl(asset) || undefined"
                      >
                        {{ officialAssetUrl(asset) }}
                      </p>
                    </div>
                  </div>
                </TableCell>
                <TableCell class="text-sm text-muted-foreground">
                  {{ asset.group_name || '素材组未知' }}
                </TableCell>
                <TableCell>
                  <div class="space-y-1">
                    <Badge
                      variant="outline"
                      :class="assetStatusClass(asset.status)"
                    >
                      {{ materialAssetStatusLabel(asset.status) }}
                    </Badge>
                    <p
                      v-if="materialAssetErrorMessage(asset)"
                      class="max-w-[220px] truncate text-[11px] text-destructive"
                      :title="materialAssetErrorMessage(asset) || ''"
                    >
                      {{ materialAssetErrorMessage(asset) }}
                    </p>
                  </div>
                </TableCell>
                <TableCell v-if="isAdmin">
                  <div class="max-w-[180px] space-y-0.5 text-xs">
                    <p class="truncate">
                      {{ asset.username || asset.user_id || '-' }}
                    </p>
                    <p class="truncate text-muted-foreground">
                      {{ asset.provider_name || asset.provider_id || '-' }}
                    </p>
                  </div>
                </TableCell>
                <TableCell class="whitespace-nowrap text-xs text-muted-foreground">
                  {{ assetMetadata(asset) }}
                </TableCell>
                <TableCell class="whitespace-nowrap text-xs text-muted-foreground">
                  {{ formatDate(asset.created_at) }}
                </TableCell>
                <TableCell class="text-right">
                  <MaterialAssetActionsMenu
                    :asset="asset"
                    :can-mutate="canMutate"
                    :deleting="deletingAssetId === asset.id"
                    @preview="openPreview"
                    @copy="copyAssetUri"
                    @use-for-video="useForVideo"
                    @rename="openRenameDialog"
                    @delete="deleteAsset"
                  />
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>

        <Pagination
          v-if="total > 0"
          :current="currentPage"
          :total="total"
          :page-size="pageSize"
          cache-key="material-assets-page-size"
          @update:current="handlePageChange"
          @update:page-size="handlePageSizeChange"
        />
      </Card>
    </div>

    <MaterialAssetPreviewDialog
      :asset="previewAsset"
      :scope="scope"
      :owner-user-id="ownerUserId"
      :show-admin-metadata="isAdmin"
      @close="previewAsset = null"
      @copy="copyAssetUri"
      @use-for-video="useForVideo"
    />

    <Dialog
      v-model:open="createGroupOpen"
      title="创建素材组"
      description="素材组会通过当前可用的方舟素材库提供商创建。"
    >
      <div class="space-y-4">
        <div v-if="isAdmin">
          <Label for="material-group-owner">所属用户 ID</Label>
          <Input
            id="material-group-owner"
            v-model="newGroupOwnerUserId"
            class="mt-2"
            placeholder="输入素材组所属用户的 ID"
            autocomplete="off"
            @keyup.enter="createMaterialGroup"
          />
          <p class="mt-1.5 text-xs leading-5 text-muted-foreground">
            创建成功后会自动切换到该用户的素材库。
          </p>
        </div>
        <div>
          <Label for="material-group-name">组名称</Label>
          <Input
            id="material-group-name"
            v-model="newGroupName"
            class="mt-2"
            placeholder="例如：人物参考素材"
            maxlength="64"
            @keyup.enter="createMaterialGroup"
          />
        </div>
        <div>
          <Label for="material-group-description">描述（可选）</Label>
          <Input
            id="material-group-description"
            v-model="newGroupDescription"
            class="mt-2"
            placeholder="说明该组素材的用途"
            maxlength="300"
          />
        </div>
      </div>
      <template #footer>
        <Button
          variant="outline"
          @click="createGroupOpen = false"
        >
          取消
        </Button>
        <Button
          :disabled="!canMutate || creatingGroup || !newGroupName.trim() || (isAdmin && !newGroupOwnerUserId.trim())"
          @click="createMaterialGroup"
        >
          <Loader2
            v-if="creatingGroup"
            class="mr-2 h-4 w-4 animate-spin"
          />
          创建素材组
        </Button>
      </template>
    </Dialog>

    <Dialog
      :open="Boolean(renamingGroup)"
      title="重命名素材组"
      description="只修改素材组名称，组内素材 URI 不会变化。"
      @update:open="(open: boolean) => { if (!open) renamingGroup = null }"
    >
      <Label for="material-group-rename">素材组名称</Label>
      <Input
        id="material-group-rename"
        v-model="groupRenameValue"
        class="mt-2"
        maxlength="64"
        @keyup.enter="renameGroup"
      />
      <template #footer>
        <Button
          variant="outline"
          @click="renamingGroup = null"
        >
          取消
        </Button>
        <Button
          :disabled="renamingGroupPending || !groupRenameValue.trim()"
          @click="renameGroup"
        >
          <Loader2
            v-if="renamingGroupPending"
            class="mr-2 h-4 w-4 animate-spin"
          />
          保存名称
        </Button>
      </template>
    </Dialog>

    <Dialog
      v-model:open="urlUploadOpen"
      title="通过公网 URL 创建素材"
      description="支持图片、视频和音频。创建为异步任务，视频通常需要更长处理时间。"
    >
      <div class="space-y-4">
        <div>
          <Label>素材类型</Label>
          <Select v-model="sourceUrlAssetType">
            <SelectTrigger class="mt-2 w-full">
              <SelectValue placeholder="选择素材类型" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Image">
                图片（Image）
              </SelectItem>
              <SelectItem value="Video">
                视频（Video）
              </SelectItem>
              <SelectItem value="Audio">
                音频（Audio）
              </SelectItem>
            </SelectContent>
          </Select>
          <p class="mt-1.5 text-xs leading-5 text-muted-foreground">
            {{ sourceUrlAssetTypeHint }}
          </p>
        </div>
        <div>
          <Label for="material-url">素材 URL</Label>
          <Input
            id="material-url"
            v-model="sourceUrl"
            class="mt-2"
            :placeholder="sourceUrlPlaceholder"
            @keyup.enter="createAssetFromUrl"
          />
        </div>
        <div>
          <Label for="material-url-name">显示名称（可选）</Label>
          <Input
            id="material-url-name"
            v-model="sourceUrlName"
            class="mt-2"
            placeholder="用于列表搜索的素材名称"
            maxlength="64"
          />
        </div>
        <div>
          <Label>素材组</Label>
          <Select v-model="sourceUrlGroupId">
            <SelectTrigger class="mt-2 w-full">
              <SelectValue placeholder="选择素材组" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem
                v-for="group in creatableGroups"
                :key="group.id"
                :value="group.id"
              >
                {{ group.name }} · {{ group.id }}
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <template #footer>
        <Button
          variant="outline"
          @click="urlUploadOpen = false"
        >
          取消
        </Button>
        <Button
          :disabled="!canCreate || creatingFromUrl || !sourceUrl.trim() || !sourceUrlGroupId"
          @click="createAssetFromUrl"
        >
          <Loader2
            v-if="creatingFromUrl"
            class="mr-2 h-4 w-4 animate-spin"
          />
          提交到方舟
        </Button>
      </template>
    </Dialog>

    <Dialog
      :open="Boolean(renamingAsset)"
      title="重命名素材"
      description="素材 ID 不会变化；Ark 返回的素材 URL 可能定期刷新。"
      @update:open="(open: boolean) => { if (!open) renamingAsset = null }"
    >
      <Label for="material-rename">素材名称</Label>
      <Input
        id="material-rename"
        v-model="renameValue"
        class="mt-2"
        maxlength="64"
        @keyup.enter="renameAsset"
      />
      <template #footer>
        <Button
          variant="outline"
          @click="renamingAsset = null"
        >
          取消
        </Button>
        <Button
          :disabled="renaming || !renameValue.trim()"
          @click="renameAsset"
        >
          <Loader2
            v-if="renaming"
            class="mr-2 h-4 w-4 animate-spin"
          />
          保存名称
        </Button>
      </template>
    </Dialog>

    <Dialog
      :open="Boolean(videoUsageAsset)"
      title="在 Seedance 请求中使用素材"
      :description="videoUsageAsset ? `${materialAssetUri(videoUsageAsset)} 已复制到剪贴板` : undefined"
      size="xl"
      @update:open="(open: boolean) => { if (!open) videoUsageAsset = null }"
    >
      <div
        v-if="videoUsageAsset"
        class="space-y-4"
      >
        <div class="rounded-lg border border-border/70 bg-muted/35 p-4">
          <p class="text-sm font-medium">
            将以下对象加入请求的 content 数组
          </p>
          <p class="mt-1 text-xs leading-5 text-muted-foreground">
            请求必须路由到与该素材相同的提供商；Aether 会校验素材所有权和提供商绑定。
          </p>
        </div>
        <pre class="max-h-80 overflow-auto rounded-lg bg-zinc-950 p-4 text-xs leading-6 text-zinc-100"><code>{{ videoUsageSnippet }}</code></pre>
      </div>
      <template #footer>
        <Button
          variant="outline"
          @click="videoUsageAsset = null"
        >
          关闭
        </Button>
        <Button @click="copyVideoUsageSnippet">
          <Copy class="mr-2 h-4 w-4" />
          复制请求片段
        </Button>
      </template>
    </Dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useDebounceFn, useIntervalFn } from '@vueuse/core'
import {
  AlertCircle,
  Copy,
  Eye,
  ExternalLink,
  File,
  Folder,
  FolderOpen,
  FolderPlus,
  Grid2X2,
  Image,
  Library,
  Link2,
  List,
  Loader2,
  MoreHorizontal,
  Music,
  Pencil,
  Search,
  Trash2,
  User,
  UserRoundCheck,
  Video,
} from 'lucide-vue-next'

import {
  createMaterialAssetsApi,
  type ArkCreatableMaterialAssetType,
  type MaterialAsset,
  type MaterialAssetGroup,
  type MaterialAssetScope,
  type MaterialAssetVerificationSession,
} from '@/api/material-assets'
import { EmptyState, LoadingState } from '@/components/common'
import { PageHeader } from '@/components/layout'
import {
  Badge,
  Button,
  Card,
  Dialog,
  Input,
  Label,
  Pagination,
  RefreshButton,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Skeleton,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useClipboard } from '@/composables/useClipboard'
import { useConfirm } from '@/composables/useConfirm'
import { useRouteQuery } from '@/composables/useRouteQuery'
import { useToast } from '@/composables/useToast'
import { useAuthStore } from '@/stores/auth'
import { formatByteSize, formatDate } from '@/utils/format'
import { parseApiError } from '@/utils/errorParser'
import MaterialAssetActionsMenu from './MaterialAssetActionsMenu.vue'
import MaterialAssetPreviewDialog from './MaterialAssetPreviewDialog.vue'
import MaterialAssetThumbnail from './MaterialAssetThumbnail.vue'
import {
  buildMaterialAssetVideoReference,
  materialAssetErrorMessage,
  materialAssetMediaType,
  materialAssetOfficialUrl,
  materialAssetRequiresVerification,
  materialAssetStatusLabel,
  materialAssetUri,
  normalizeMaterialAssetStatus,
} from '@/features/material-assets/utils/materialAssetPresentation'

const props = defineProps<{
  scope: MaterialAssetScope
}>()

const authStore = useAuthStore()
const { toast } = useToast()
const { copyToClipboard } = useClipboard()
const { confirmDanger } = useConfirm()
const { getQueryValue, patchQuery } = useRouteQuery()
const api = createMaterialAssetsApi(props.scope)

const isAdmin = computed(() => props.scope === 'admin')
const canMutate = computed(() => !isAdmin.value || authStore.canOperateAdmin)
const ownerUserId = computed(() => isAdmin.value ? appliedAdminUserId.value.trim() : '')
const hasUnappliedAdminOwner = computed(() => isAdmin.value
  && adminUserId.value.trim() !== appliedAdminUserId.value.trim())
const canCreate = computed(() => canMutate.value && (!isAdmin.value || (
  Boolean(ownerUserId.value) && !hasUnappliedAdminOwner.value
)))

const groups = ref<MaterialAssetGroup[]>([])
const assets = ref<MaterialAsset[]>([])
const total = ref(0)
const currentPage = ref(Number.parseInt(getQueryValue('page') || '1', 10) || 1)
const pageSize = ref(20)
const selectedGroupId = ref(getQueryValue('group') || '')
const searchQuery = ref(getQueryValue('search') || '')
const filterType = ref(getQueryValue('type') || 'all')
const filterStatus = ref(getQueryValue('status') || 'all')
const adminUserId = ref(getQueryValue('user_id') || '')
const appliedAdminUserId = ref(adminUserId.value)
const viewMode = ref<'grid' | 'list'>(getQueryValue('view') === 'list' ? 'list' : 'grid')
const loading = ref(false)
const groupsLoading = ref(false)
const loadError = ref<string | null>(null)
let assetRequestSequence = 0
let groupRequestSequence = 0
let activeForegroundRequests = 0

const createGroupOpen = ref(false)
const newGroupOwnerUserId = ref('')
const newGroupName = ref('')
const newGroupDescription = ref('')
const creatingGroup = ref(false)
const renamingGroup = ref<MaterialAssetGroup | null>(null)
const groupRenameValue = ref('')
const renamingGroupPending = ref(false)
const deletingGroupId = ref<string | null>(null)

const urlUploadOpen = ref(false)
const sourceUrl = ref('')
const sourceUrlName = ref('')
const sourceUrlGroupId = ref('')
const sourceUrlAssetType = ref<ArkCreatableMaterialAssetType>('Image')
const creatingFromUrl = ref(false)

const previewAsset = ref<MaterialAsset | null>(null)
const renamingAsset = ref<MaterialAsset | null>(null)
const renameValue = ref('')
const renaming = ref(false)
const deletingAssetId = ref<string | null>(null)
const verificationSessionPending = ref(false)
const activeVerificationSession = ref<MaterialAssetVerificationSession | null>(null)
let verificationSessionGeneration = 0
const videoUsageAsset = ref<MaterialAsset | null>(null)

const activeGroupName = computed(() => {
  if (!selectedGroupId.value) return '全部素材'
  return groups.value.find(group => group.id === selectedGroupId.value)?.name || '素材组'
})

const creatableGroups = computed(() => groups.value.filter(
  group => group.group_type.trim().toLowerCase() === 'aigc',
))

const hasActiveFilters = computed(() => Boolean(
  selectedGroupId.value
  || searchQuery.value.trim()
  || filterType.value !== 'all'
  || filterStatus.value !== 'all'
  || (isAdmin.value && appliedAdminUserId.value.trim()),
))

const emptyActionText = computed(() => {
  if (hasActiveFilters.value || !canCreate.value) return undefined
  return creatableGroups.value.length === 0 ? '先创建 AIGC 素材组' : '通过 URL 添加素材'
})

const emptyStateDescription = computed(() => {
  if (hasActiveFilters.value) return '调整筛选条件后重试'
  if (creatableGroups.value.length === 0) return '创建 AIGC 素材组后，可通过公网 URL 添加素材'
  return '可通过公网 URL 创建首个素材'
})

const verificationSessionStatus = computed(() => (
  activeVerificationSession.value?.status?.trim().toLowerCase() || 'pending'
))
const verificationSessionIsPending = computed(() => ![
  'active',
  'completed',
  'succeeded',
  'success',
  'failed',
  'expired',
  'cancelled',
  'canceled',
  'rejected',
].includes(verificationSessionStatus.value))
const hasPendingVerificationSession = computed(() => Boolean(
  activeVerificationSession.value && verificationSessionIsPending.value,
))
const verificationSessionUrl = computed(() => safeHttpUrl(
  activeVerificationSession.value?.h5_link || activeVerificationSession.value?.verification_url,
))
const verificationSessionTitle = computed(() => {
  if (verificationSessionIsPending.value) return '等待完成真人验证'
  if (['active', 'completed', 'succeeded', 'success'].includes(verificationSessionStatus.value)) {
    return '真人验证已完成'
  }
  return '真人验证未完成'
})
const verificationSessionDescription = computed(() => {
  const session = activeVerificationSession.value
  if (!session) return ''
  if (verificationSessionIsPending.value) return '请在验证页完成人脸与身份核验，本页会自动同步结果。'
  if (session.group_id) return `已生成 LivenessFace 素材组 ${session.group_id}`
  return session.error_message || `会话状态：${session.status}`
})

const pageStatusCounts = computed(() => assets.value.reduce((counts, asset) => {
  counts[normalizeMaterialAssetStatus(asset.status)] += 1
  return counts
}, { active: 0, processing: 0, failed: 0 }))

const hasProcessingAssets = computed(() => assets.value.some(
  asset => normalizeMaterialAssetStatus(asset.status) === 'processing',
))

const videoUsageSnippet = computed(() => {
  if (!videoUsageAsset.value) return ''
  return JSON.stringify(buildMaterialAssetVideoReference(videoUsageAsset.value), null, 2)
})

const sourceUrlPlaceholder = computed(() => ({
  Image: 'https://example.com/reference.jpg',
  Video: 'https://example.com/reference.mp4',
  Audio: 'https://example.com/reference.mp3',
})[sourceUrlAssetType.value])

const sourceUrlAssetTypeHint = computed(() => ({
  Image: '支持 jpeg、png、webp、bmp、tiff、gif、heic、heif，单张小于 30 MB。',
  Video: '支持 mp4、mov，时长 2–30 秒，单个文件不超过 200 MB。',
  Audio: '支持 wav、mp3，时长 2–30 秒，单个文件不超过 15 MB。',
})[sourceUrlAssetType.value])

function syncRouteQuery() {
  patchQuery({
    group: selectedGroupId.value || undefined,
    search: searchQuery.value.trim() || undefined,
    type: filterType.value === 'all' ? undefined : filterType.value,
    status: filterStatus.value === 'all' ? undefined : filterStatus.value,
    user_id: isAdmin.value ? appliedAdminUserId.value.trim() || undefined : undefined,
    view: viewMode.value === 'grid' ? undefined : viewMode.value,
    page: currentPage.value > 1 ? String(currentPage.value) : undefined,
  })
}

async function fetchGroups() {
  const requestSequence = ++groupRequestSequence
  groupsLoading.value = true
  try {
    const response = await api.listGroups({
      user_id: isAdmin.value ? appliedAdminUserId.value.trim() || undefined : undefined,
    })
    if (requestSequence !== groupRequestSequence) return
    groups.value = response.items
  } catch (error: unknown) {
    if (requestSequence !== groupRequestSequence) return
    toast({
      title: '获取素材组失败',
      description: parseApiError(error, '获取素材组失败'),
      variant: 'destructive',
    })
  } finally {
    if (requestSequence === groupRequestSequence) groupsLoading.value = false
  }
}

async function fetchAssets(background = false) {
  const requestSequence = ++assetRequestSequence
  if (!background) {
    activeForegroundRequests += 1
    loading.value = true
    loadError.value = null
  }
  if (!background) syncRouteQuery()
  try {
    const response = await api.listAssets({
      group_id: selectedGroupId.value || undefined,
      search: searchQuery.value.trim() || undefined,
      type: filterType.value === 'all' ? undefined : filterType.value,
      status: filterStatus.value === 'all' ? undefined : filterStatus.value,
      user_id: isAdmin.value ? appliedAdminUserId.value.trim() || undefined : undefined,
      page: currentPage.value,
      page_size: pageSize.value,
    })
    if (requestSequence !== assetRequestSequence) return
    assets.value = response.items
    total.value = response.total
  } catch (error: unknown) {
    if (requestSequence !== assetRequestSequence) return
    if (!background) loadError.value = parseApiError(error, '获取素材列表失败')
  } finally {
    if (!background) {
      activeForegroundRequests = Math.max(0, activeForegroundRequests - 1)
      loading.value = activeForegroundRequests > 0
    }
  }
}

async function refresh() {
  await Promise.all([fetchGroups(), fetchAssets()])
}

function selectGroup(groupId: string) {
  if (selectedGroupId.value === groupId) return
  selectedGroupId.value = groupId
}

function handleEmptyAction() {
  if (!canCreate.value) return
  if (creatableGroups.value.length === 0) createGroupOpen.value = true
  else openUrlUploadDialog()
}

function createActionTitle(defaultTitle: string): string {
  if (!canMutate.value) return '当前账号没有素材库写权限'
  if (isAdmin.value && !ownerUserId.value) return '请先填写并应用用户 ID'
  if (hasUnappliedAdminOwner.value) return '请先应用当前用户 ID'
  return defaultTitle
}

function createGroupActionTitle(): string {
  if (!canMutate.value) return '当前账号没有素材库写权限'
  if (isAdmin.value && (!ownerUserId.value || hasUnappliedAdminOwner.value)) {
    return '创建素材组并指定归属用户'
  }
  return '创建素材组'
}

function openCreateGroupDialog() {
  if (!canMutate.value) return
  newGroupOwnerUserId.value = ownerUserId.value || adminUserId.value.trim()
  createGroupOpen.value = true
}

function applySearchNow() {
  currentPage.value = 1
  fetchAssets()
}

function applyAdminUserFilter() {
  verificationSessionGeneration += 1
  pauseVerificationPoll()
  activeVerificationSession.value = null
  appliedAdminUserId.value = adminUserId.value.trim()
  selectedGroupId.value = ''
  currentPage.value = 1
  groups.value = []
  assets.value = []
  total.value = 0
  refresh()
}

function handlePageChange(page: number) {
  currentPage.value = page
  fetchAssets()
}

function handlePageSizeChange(size: number) {
  pageSize.value = size
  currentPage.value = 1
  fetchAssets()
}

function normalizedMediaType(asset: MaterialAsset): string {
  return materialAssetMediaType(asset)
}

function officialAssetUrl(asset: MaterialAsset): string | null {
  return materialAssetOfficialUrl(asset)
}

function mediaIcon(mediaType: string | null | undefined) {
  switch (mediaType?.trim().toLowerCase()) {
    case 'image':
      return Image
    case 'video':
      return Video
    case 'audio':
      return Music
    default:
      return File
  }
}

function assetStatusClass(status: string): string {
  const normalized = normalizeMaterialAssetStatus(status)
  if (normalized === 'active') return 'border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
  if (normalized === 'failed') return 'border-destructive/30 bg-destructive/10 text-destructive'
  return 'border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400'
}

function assetMetadata(asset: MaterialAsset): string {
  const parts: string[] = []
  if (asset.width && asset.height) parts.push(`${asset.width}×${asset.height}`)
  if (asset.duration_seconds) parts.push(`${asset.duration_seconds}s`)
  if (asset.size_bytes !== null && asset.size_bytes !== undefined) parts.push(formatByteSize(asset.size_bytes))
  return parts.join(' · ') || asset.mime_type || '-'
}

function openPreview(asset: MaterialAsset) {
  if (normalizeMaterialAssetStatus(asset.status) !== 'active') return
  previewAsset.value = asset
}

async function copyAssetUri(asset: MaterialAsset) {
  await copyToClipboard(materialAssetUri(asset))
}

async function useForVideo(asset: MaterialAsset) {
  const uri = materialAssetUri(asset)
  const copied = await copyToClipboard(uri, false)
  if (!copied) return
  previewAsset.value = null
  videoUsageAsset.value = asset
  toast({
    title: '素材 URI 已复制',
    description: '已展示 Seedance content 对象，可直接复制到请求中。',
  })
}

async function copyVideoUsageSnippet() {
  if (!videoUsageSnippet.value) return
  await copyToClipboard(videoUsageSnippet.value)
}

async function createMaterialGroup() {
  const name = newGroupName.value.trim()
  const targetUserId = isAdmin.value ? newGroupOwnerUserId.value.trim() : ''
  if (!canMutate.value || !name || (isAdmin.value && !targetUserId) || creatingGroup.value) return
  creatingGroup.value = true
  try {
    const group = await api.createGroup({
      name,
      description: newGroupDescription.value.trim() || undefined,
      group_type: 'AIGC',
      user_id: targetUserId || undefined,
    })
    if (isAdmin.value) {
      adminUserId.value = targetUserId
      appliedAdminUserId.value = targetUserId
    }
    createGroupOpen.value = false
    newGroupOwnerUserId.value = ''
    newGroupName.value = ''
    newGroupDescription.value = ''
    selectedGroupId.value = group.id
    toast({ title: '素材组已创建', description: group.name })
    await refresh()
  } catch (error: unknown) {
    toast({
      title: '创建素材组失败',
      description: parseApiError(error, '创建素材组失败'),
      variant: 'destructive',
    })
  } finally {
    creatingGroup.value = false
  }
}

function openRenameGroupDialog(group: MaterialAssetGroup) {
  if (!canMutate.value) return
  renamingGroup.value = group
  groupRenameValue.value = group.name
}

async function renameGroup() {
  const group = renamingGroup.value
  const name = groupRenameValue.value.trim()
  if (!group || !name || renamingGroupPending.value) return
  renamingGroupPending.value = true
  try {
    const updated = await api.renameGroup(group.id, name, ownerUserId.value)
    groups.value = groups.value.map(item => item.id === updated.id ? updated : item)
    renamingGroup.value = null
    toast({ title: '素材组已重命名' })
  } catch (error: unknown) {
    toast({
      title: '重命名素材组失败',
      description: parseApiError(error, '重命名素材组失败'),
      variant: 'destructive',
    })
  } finally {
    renamingGroupPending.value = false
  }
}

async function deleteGroup(group: MaterialAssetGroup) {
  if (!canMutate.value || deletingGroupId.value) return
  const impact = group.asset_count > 0
    ? `该操作会同时永久删除组内 ${group.asset_count} 项素材`
    : '删除后无法恢复'
  const confirmed = await confirmDanger(
    `确定删除素材组“${group.name}”吗？${impact}。`,
    '删除素材组',
    '删除素材组',
  )
  if (!confirmed) return
  if (group.asset_count > 0) {
    const cascadeConfirmed = await confirmDanger(
      `最后确认：将同时永久删除“${group.name}”中的 ${group.asset_count} 项素材，相关 asset:// URI 会立即失效。`,
      '确认级联删除',
      '永久删除组及素材',
    )
    if (!cascadeConfirmed) return
  }

  deletingGroupId.value = group.id
  try {
    await api.deleteGroup(group.id, ownerUserId.value)
    if (selectedGroupId.value === group.id) selectedGroupId.value = ''
    toast({ title: '素材组已删除' })
    await refresh()
  } catch (error: unknown) {
    toast({
      title: '删除素材组失败',
      description: parseApiError(error, '删除素材组失败'),
      variant: 'destructive',
    })
  } finally {
    deletingGroupId.value = null
  }
}

function openUrlUploadDialog() {
  if (!canCreate.value) return
  if (creatableGroups.value.length === 0) {
    createGroupOpen.value = true
    return
  }
  sourceUrl.value = ''
  sourceUrlName.value = ''
  sourceUrlAssetType.value = 'Image'
  sourceUrlGroupId.value = creatableGroups.value.some(group => group.id === selectedGroupId.value)
    ? selectedGroupId.value
    : creatableGroups.value[0]?.id || ''
  urlUploadOpen.value = true
}

async function createAssetFromUrl() {
  if (!canCreate.value || creatingFromUrl.value) return
  if (!sourceUrlGroupId.value) {
    toast({ title: '请选择素材组', variant: 'destructive' })
    return
  }
  const url = sourceUrl.value.trim()
  try {
    const parsed = new URL(url)
    if (parsed.protocol !== 'https:') throw new Error('unsupported protocol')
  } catch {
    toast({ title: 'URL 格式错误', description: '请输入可从公网访问的 HTTPS URL', variant: 'destructive' })
    return
  }

  creatingFromUrl.value = true
  try {
    const asset = await api.createFromUrl({
      url,
      name: sourceUrlName.value.trim() || undefined,
      group_id: sourceUrlGroupId.value,
      asset_type: sourceUrlAssetType.value,
      user_id: ownerUserId.value || undefined,
    })
    urlUploadOpen.value = false
    toast({ title: '素材已进入处理队列', description: asset.name })
    currentPage.value = 1
    await refresh()
  } catch (error: unknown) {
    toast({
      title: 'URL 素材创建失败',
      description: parseApiError(error, 'URL 素材创建失败'),
      variant: 'destructive',
    })
  } finally {
    creatingFromUrl.value = false
  }
}

function openRenameDialog(asset: MaterialAsset) {
  if (!canMutate.value) return
  renamingAsset.value = asset
  renameValue.value = asset.name
}

async function renameAsset() {
  const asset = renamingAsset.value
  const name = renameValue.value.trim()
  if (!asset || !name || renaming.value) return
  renaming.value = true
  try {
    const updated = await api.renameAsset(asset.id, { name }, ownerUserId.value)
    assets.value = assets.value.map(item => item.id === updated.id ? updated : item)
    if (previewAsset.value?.id === updated.id) previewAsset.value = updated
    renamingAsset.value = null
    toast({ title: '素材已重命名' })
  } catch (error: unknown) {
    toast({
      title: '重命名失败',
      description: parseApiError(error, '重命名失败'),
      variant: 'destructive',
    })
  } finally {
    renaming.value = false
  }
}

async function deleteAsset(asset: MaterialAsset) {
  if (!canMutate.value || deletingAssetId.value) return
  const confirmed = await confirmDanger(
    `确定删除素材“${asset.name}”吗？删除后 asset:// URI 将不可再用于视频生成。`,
    '删除素材',
    '删除素材',
  )
  if (!confirmed) return

  deletingAssetId.value = asset.id
  try {
    await api.deleteAsset(asset.id, ownerUserId.value)
    if (previewAsset.value?.id === asset.id) previewAsset.value = null
    toast({ title: '素材已删除' })
    await refresh()
  } catch (error: unknown) {
    toast({
      title: '删除素材失败',
      description: parseApiError(error, '删除素材失败'),
      variant: 'destructive',
    })
  } finally {
    deletingAssetId.value = null
  }
}

async function startVerification() {
  if (!canCreate.value || verificationSessionPending.value || hasPendingVerificationSession.value) return
  const sessionGeneration = ++verificationSessionGeneration
  const ownerAtStart = ownerUserId.value
  pauseVerificationPoll()
  activeVerificationSession.value = null
  verificationSessionPending.value = true
  const verificationWindow = window.open('about:blank', '_blank')
  if (verificationWindow) verificationWindow.opener = null

  try {
    const callbackUrl = new URL(window.location.href)
    callbackUrl.hash = ''
    const session = await api.createVerificationSession({
      callback_url: callbackUrl.toString(),
      user_id: ownerAtStart || undefined,
    })
    if (sessionGeneration !== verificationSessionGeneration || ownerAtStart !== ownerUserId.value) {
      verificationWindow?.close()
      return
    }
    activeVerificationSession.value = session
    const verificationUrl = safeHttpUrl(session.h5_link || session.verification_url)
    if (verificationUrl && verificationWindow) {
      verificationWindow.location.replace(verificationUrl)
      toast({ title: '真人验证会话已创建', description: '请在新窗口完成验证' })
    } else if (verificationUrl) {
      toast({ title: '真人验证会话已创建', description: '浏览器阻止了新窗口，请点击下方按钮打开验证页' })
    } else {
      verificationWindow?.close()
      toast({ title: '真人验证会话已创建', description: `会话状态：${session.status}` })
    }
    if (verificationSessionIsPending.value) resumeVerificationPoll()
    else await applyVerificationSessionResult(session, sessionGeneration)
  } catch (error: unknown) {
    verificationWindow?.close()
    toast({
      title: '创建真人验证会话失败',
      description: parseApiError(error, '创建真人验证会话失败'),
      variant: 'destructive',
    })
  } finally {
    verificationSessionPending.value = false
  }
}

function reopenVerificationSession() {
  if (!verificationSessionUrl.value) return
  window.open(verificationSessionUrl.value, '_blank', 'noopener,noreferrer')
}

function safeHttpUrl(value: string | null | undefined): string | null {
  if (!value) return null
  try {
    const parsed = new URL(value)
    return ['http:', 'https:'].includes(parsed.protocol) ? parsed.toString() : null
  } catch {
    return null
  }
}

let verificationPollInFlight = false
async function pollVerificationSession() {
  const session = activeVerificationSession.value
  if (!session || verificationPollInFlight || document.hidden) return
  const sessionGeneration = verificationSessionGeneration

  if (session.expires_at) {
    const expiresAt = verificationExpiryMillis(session.expires_at)
    if (Number.isFinite(expiresAt) && expiresAt <= Date.now()) {
      activeVerificationSession.value = { ...session, status: 'Expired' }
      pauseVerificationPoll()
      return
    }
  }

  verificationPollInFlight = true
  try {
    const updated = await api.getVerificationSession(session.id, ownerUserId.value)
    if (
      sessionGeneration !== verificationSessionGeneration
      || activeVerificationSession.value?.id !== session.id
    ) return
    await applyVerificationSessionResult(updated, sessionGeneration)
  } catch {
    // Verification polling is best-effort; the regular refresh action remains available.
  } finally {
    verificationPollInFlight = false
  }
}

function verificationExpiryMillis(value: string | number): number {
  const numeric = typeof value === 'number' ? value : Number(value)
  if (Number.isFinite(numeric)) return numeric < 1_000_000_000_000 ? numeric * 1_000 : numeric
  return typeof value === 'string' ? Date.parse(value) : Number.NaN
}

async function applyVerificationSessionResult(
  session: MaterialAssetVerificationSession,
  sessionGeneration = verificationSessionGeneration,
) {
  if (sessionGeneration !== verificationSessionGeneration) return
  activeVerificationSession.value = session
  const status = session.status.trim().toLowerCase()
  if (['active', 'completed', 'succeeded', 'success'].includes(status)) {
    pauseVerificationPoll()
    await fetchGroups()
    if (
      sessionGeneration !== verificationSessionGeneration
      || activeVerificationSession.value?.id !== session.id
    ) return
    if (session.group_id) selectedGroupId.value = session.group_id
    currentPage.value = 1
    await fetchAssets()
    if (
      sessionGeneration !== verificationSessionGeneration
      || activeVerificationSession.value?.id !== session.id
    ) return
    toast({
      title: '真人验证已完成',
      description: session.group_id ? `已创建素材组：${session.group_id}` : '真人素材组已同步',
    })
    return
  }

  if (['failed', 'expired', 'cancelled', 'canceled', 'rejected'].includes(status)) {
    pauseVerificationPoll()
    toast({
      title: '真人验证未完成',
      description: session.error_message || `会话状态：${session.status}`,
      variant: 'destructive',
    })
  }
}

const debouncedSearch = useDebounceFn(() => {
  currentPage.value = 1
  fetchAssets()
}, 300)

let processingPollInFlight = false
const { pause: pauseProcessingPoll, resume: resumeProcessingPoll } = useIntervalFn(async () => {
  if (document.hidden || processingPollInFlight) return
  processingPollInFlight = true
  try {
    await fetchAssets(true)
  } finally {
    processingPollInFlight = false
  }
}, 5_000, { immediate: false })

const { pause: pauseVerificationPoll, resume: resumeVerificationPoll } = useIntervalFn(() => {
  void pollVerificationSession()
}, 4_000, { immediate: false })

watch(searchQuery, () => {
  debouncedSearch()
})

watch([selectedGroupId, filterType, filterStatus], () => {
  currentPage.value = 1
  fetchAssets()
})

watch(viewMode, () => {
  syncRouteQuery()
})

watch(hasProcessingAssets, (hasProcessing) => {
  if (hasProcessing) resumeProcessingPoll()
  else pauseProcessingPoll()
}, { immediate: true })

onMounted(() => {
  refresh()
})

onBeforeUnmount(() => {
  assetRequestSequence += 1
  groupRequestSequence += 1
  verificationSessionGeneration += 1
  pauseProcessingPoll()
  pauseVerificationPoll()
})
</script>
